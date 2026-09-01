"""Reconcile tracked live Spot positions with their native Binance OCO exits."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import TradingExecutionMode
from app.core.logging import get_logger
from app.models.portfolio import Position
from app.services.exchange.binance_client import binance_client
from app.services.exchange.live_execution_readiness import live_protection_readiness
from app.services.exchange.live_trading_guard import live_trading_guard
from app.services.exchange.spot_account_inventory_reconciler import (
    SpotAccountInventoryReconciler,
)
from app.services.portfolio.tracker import portfolio_tracker

logger = get_logger(__name__)


class SpotProtectionExchange(Protocol):
    async def get_open_spot_order_lists(self) -> list[dict[str, Any]]: ...

    async def get_spot_order_list(self, order_list_id: int) -> dict[str, Any]: ...

    async def get_order_status(self, symbol: str, order_id: int) -> dict[str, Any]: ...

    async def get_account(self) -> dict[str, Any]: ...

    async def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class ProtectionReconciliationReport:
    """Public-safe summary; it intentionally excludes exchange credentials and payloads."""

    checked_positions: int
    active_protections: int
    closed_positions: int
    issues: tuple[str, ...] = ()
    checked_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def ready(self) -> bool:
        return not self.issues

    def as_dict(self, max_age_seconds: int) -> dict[str, object]:
        readiness = live_protection_readiness.status(max_age_seconds)
        return {
            **readiness,
            "checked_positions": self.checked_positions,
            "active_protections": self.active_protections,
            "closed_positions": self.closed_positions,
            "issues": list(self.issues),
        }


class SpotProtectionReconciler:
    """Fail closed on every difference between the local ledger and Binance OCOs."""

    def __init__(self, exchange: SpotProtectionExchange | None = None) -> None:
        self._exchange = exchange or binance_client
        self._inventory_reconciler = SpotAccountInventoryReconciler(self._exchange)

    async def reconcile(
        self,
        db: AsyncSession,
        *,
        execution_mode: TradingExecutionMode = TradingExecutionMode.LIVE,
    ) -> ProtectionReconciliationReport:
        """Synchronize OPEN exchange Spot positions without modifying Binance orders."""
        checked_at = datetime.now(UTC)
        result = await db.execute(
            select(Position).where(
                Position.is_open.is_(True),
                Position.execution_mode == execution_mode.value,
            ).with_for_update()
        )
        positions = list(result.scalars().all())
        issues: list[str] = []
        active_protections = 0
        closed_positions = 0

        try:
            exchange_open_lists = await self._exchange.get_open_spot_order_lists()
        except Exception as exc:
            issue = f"could not read Binance Spot OCO order lists: {exc}"
            if execution_mode == TradingExecutionMode.LIVE:
                live_protection_readiness.mark_error(issue, checked_at)
                live_trading_guard.disarm("native protection reconciliation failed")
            logger.critical("spot_protection_reconciliation_failed", error=str(exc), exc_info=True)
            raise

        active_by_id = {
            _order_list_id(order_list): order_list
            for order_list in exchange_open_lists
            if _order_list_id(order_list) is not None
        }
        expected_ids = {
            int(position.protective_order_list_id)
            for position in positions
            if position.protective_order_list_id is not None
        }

        for order_list in exchange_open_lists:
            order_list_id = _order_list_id(order_list)
            if (
                order_list_id is not None
                and order_list_id not in expected_ids
                and str(order_list.get("listClientOrderId", "")).startswith("TM-P-")
            ):
                issues.append(f"orphaned bot OCO order list {order_list_id} is open on Binance")

        for position in positions:
            protected = await self._reconcile_position(
                db=db,
                position=position,
                active_order_list=(
                    active_by_id.get(int(position.protective_order_list_id))
                    if position.protective_order_list_id is not None
                    else None
                ),
                checked_at=checked_at,
                issues=issues,
            )
            if protected == "ACTIVE":
                active_protections += 1
            elif protected == "CLOSED":
                closed_positions += 1

        # Testnet accounts start with a broad set of faucet balances. They are
        # not evidence of an untracked TradeMaster position. The strict full
        # account-inventory comparison is a LIVE-only guard; Testnet still
        # reconciles every tracked position and TradeMaster-owned OCO order.
        if execution_mode == TradingExecutionMode.LIVE:
            try:
                inventory_report = await self._inventory_reconciler.reconcile(
                    positions,
                    execution_mode=execution_mode.value,
                )
                issues.extend(inventory_report.issues)
            except Exception as exc:
                issues.append(f"could not reconcile Binance Spot account inventory: {exc}")

        await db.flush()
        report = ProtectionReconciliationReport(
            checked_positions=len(positions),
            active_protections=active_protections,
            closed_positions=closed_positions,
            issues=tuple(issues),
            checked_at=checked_at,
        )
        if report.ready and execution_mode == TradingExecutionMode.LIVE:
            live_protection_readiness.mark_ready(checked_at)
            logger.info(
                "spot_protection_reconciled",
                checked_positions=report.checked_positions,
                active_protections=report.active_protections,
                closed_positions=report.closed_positions,
            )
        elif execution_mode == TradingExecutionMode.LIVE:
            live_protection_readiness.mark_unresolved(issues, checked_at)
            live_trading_guard.disarm("native protection reconciliation found unresolved state")
            logger.critical(
                "spot_protection_reconciliation_unresolved",
                checked_positions=report.checked_positions,
                issues=issues,
            )
        elif report.ready:
            logger.info(
                "spot_protection_reconciled",
                execution_mode=execution_mode.value,
                checked_positions=report.checked_positions,
                active_protections=report.active_protections,
                closed_positions=report.closed_positions,
            )
        else:
            logger.error(
                "spot_protection_reconciliation_unresolved",
                execution_mode=execution_mode.value,
                checked_positions=report.checked_positions,
                issues=issues,
            )
        return report

    async def _reconcile_position(
        self,
        *,
        db: AsyncSession,
        position: Position,
        active_order_list: dict[str, Any] | None,
        checked_at: datetime,
        issues: list[str],
    ) -> str:
        order_list_id = position.protective_order_list_id
        protected_quantity = _decimal(position.protective_quantity)
        if order_list_id is None or protected_quantity <= 0:
            self._mark_missing(position, checked_at)
            issues.append(f"position {position.id} has no complete native OCO reference")
            return "ISSUE"
        if protected_quantity != _decimal(position.quantity):
            self._mark_missing(position, checked_at)
            issues.append(
                f"position {position.id} native OCO quantity does not cover the tracked position"
            )
            return "ISSUE"

        if active_order_list is not None:
            if _matches_active_oco(position, active_order_list):
                position.protection_status = "ACTIVE"
                position.protection_updated_at = checked_at
                return "ACTIVE"
            self._mark_missing(position, checked_at)
            issues.append(f"position {position.id} has an invalid active Binance OCO state")
            return "ISSUE"

        try:
            terminal_list = await self._exchange.get_spot_order_list(int(order_list_id))
        except Exception as exc:
            self._mark_missing(position, checked_at)
            issues.append(
                f"position {position.id} OCO {order_list_id} is absent or cannot be queried: {exc}"
            )
            return "ISSUE"

        terminal_state = await self._resolve_terminal_oco(
            position=position,
            terminal_list=terminal_list,
            protected_quantity=protected_quantity,
        )
        if terminal_state is None:
            self._mark_missing(position, checked_at)
            issues.append(
                f"position {position.id} OCO {order_list_id} closed without a full "
                "confirmed sell fill"
            )
            return "ISSUE"

        await portfolio_tracker.close_position(db, position, terminal_state)
        position.protection_status = "EXIT_FILLED"
        position.protection_updated_at = checked_at
        return "CLOSED"

    async def _resolve_terminal_oco(
        self,
        *,
        position: Position,
        terminal_list: dict[str, Any],
        protected_quantity: Decimal,
    ) -> float | None:
        if (
            _order_list_id(terminal_list) != position.protective_order_list_id
            or terminal_list.get("contingencyType") != "OCO"
            or terminal_list.get("listStatusType") != "ALL_DONE"
            or terminal_list.get("listOrderStatus") != "ALL_DONE"
        ):
            return None

        order_ids = _child_order_ids(terminal_list)
        if len(order_ids) < 2:
            return None
        try:
            child_orders = [
                await self._exchange.get_order_status(position.symbol, order_id)
                for order_id in order_ids
            ]
        except Exception as exc:
            logger.warning(
                "spot_protection_terminal_order_lookup_failed",
                position_id=position.id,
                order_list_id=position.protective_order_list_id,
                error=str(exc),
            )
            return None

        filled_sells = [
            order
            for order in child_orders
            if order.get("side") == "SELL" and order.get("status") == "FILLED"
        ]
        if len(filled_sells) != 1:
            return None
        exit_order = filled_sells[0]
        if int(exit_order.get("orderListId", -1)) != position.protective_order_list_id:
            return None
        exit_quantity = _decimal(exit_order.get("executedQty"))
        if exit_quantity != protected_quantity:
            return None
        exit_price = _average_fill_price(exit_order)
        return float(exit_price) if exit_price is not None else None

    @staticmethod
    def _mark_missing(position: Position, checked_at: datetime) -> None:
        position.protection_status = "MISSING"
        position.protection_updated_at = checked_at


def _order_list_id(order_list: dict[str, Any]) -> int | None:
    try:
        return int(order_list["orderListId"])
    except (KeyError, TypeError, ValueError):
        return None


def _child_order_ids(order_list: dict[str, Any]) -> list[int]:
    order_ids: list[int] = []
    for order in order_list.get("orders", []):
        try:
            order_ids.append(int(order["orderId"]))
        except (KeyError, TypeError, ValueError):
            return []
    return order_ids


def _matches_active_oco(position: Position, order_list: dict[str, Any]) -> bool:
    return (
        _order_list_id(order_list) == position.protective_order_list_id
        and order_list.get("symbol") == position.symbol
        and order_list.get("contingencyType") == "OCO"
        and order_list.get("listStatusType") == "EXEC_STARTED"
        and order_list.get("listOrderStatus") == "EXECUTING"
    )


def _average_fill_price(order: dict[str, Any]) -> Decimal | None:
    quantity = _decimal(order.get("executedQty"))
    quote = _decimal(order.get("cummulativeQuoteQty"))
    if quantity > 0 and quote > 0:
        return quote / quantity
    price = _decimal(order.get("price"))
    return price if price > 0 else None


def _decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value or 0))
    except Exception:
        return Decimal("0")


spot_protection_reconciler = SpotProtectionReconciler()
