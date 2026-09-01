"""Controlled manual exits for protected Binance Spot LIVE positions."""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal, Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import TradingExecutionMode
from app.core.logging import get_logger
from app.models.portfolio import Position
from app.models.trade import Order, OrderStatus, OrderType
from app.services.exchange.binance_client import binance_client
from app.services.exchange.live_trading_guard import live_trading_guard
from app.services.portfolio.tracker import portfolio_tracker

logger = get_logger(__name__)


class SpotPositionCloseError(Exception):
    """Raised when a protected Spot position cannot be closed safely."""


class SpotPositionCloseExchange(Protocol):
    async def cancel_spot_order_list(self, symbol: str, order_list_id: int) -> dict[str, Any]: ...

    async def get_spot_order_list(self, order_list_id: int) -> dict[str, Any]: ...

    async def get_order_status(self, symbol: str, order_id: int) -> dict[str, Any]: ...

    async def place_market_order(
        self, symbol: str, side: str, quantity: Decimal, client_order_id: str
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ExchangePositionCloseReport:
    status: Literal["OCO_FILLED", "MANUAL_MARKET_FILLED", "STRATEGY_MARKET_FILLED"]
    position_id: int
    symbol: str
    exit_order_id: str
    exit_price: Decimal
    closed_at: datetime

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "status": self.status,
            "position_id": self.position_id,
            "symbol": self.symbol,
            "exit_order_id": self.exit_order_id,
            "exit_price": float(self.exit_price),
            "closed_at": self.closed_at.isoformat(),
        }


@dataclass(frozen=True)
class _TerminalOcoState:
    kind: Literal["CANCELLED", "SELL_FILLED", "AMBIGUOUS"]
    exit_order: dict[str, Any] | None = None


class SpotPositionCloser:
    """Cancel the exact OCO and sell once, never virtually close a Spot position."""

    def __init__(self, exchange: SpotPositionCloseExchange | None = None) -> None:
        self._exchange = exchange or binance_client

    async def close(
        self,
        *,
        db: AsyncSession,
        position_id: int,
        totp_code: str,
    ) -> ExchangePositionCloseReport:
        live_trading_guard.require_live_exit(totp_code)
        position = await self._load_protected_position(
            db,
            position_id,
            execution_mode=TradingExecutionMode.LIVE,
        )

        live_trading_guard.disarm("operator requested a live Spot position exit")
        return await self._close_protected_position(
            db=db,
            position=position,
            market_exit_status="MANUAL_MARKET_FILLED",
            market_exit_notes=f"Operator-requested Spot exit for position #{position.id}",
        )

    async def close_for_strategy(
        self,
        *,
        db: AsyncSession,
        position_id: int,
        execution_mode: TradingExecutionMode,
    ) -> ExchangePositionCloseReport:
        """Close an exact, OCO-protected Spot long after a deployed strategy exit.

        Testnet has no real capital. LIVE remains gated by the short-lived
        arming session; in both modes a direct naked SELL is prohibited.
        """
        if execution_mode not in {TradingExecutionMode.TESTNET, TradingExecutionMode.LIVE}:
            raise SpotPositionCloseError("strategy exchange exits require TESTNET or LIVE mode")
        if execution_mode == TradingExecutionMode.LIVE:
            live_trading_guard.require_live_strategy_exit()
        position = await self._load_protected_position(
            db,
            position_id,
            execution_mode=execution_mode,
        )
        return await self._close_protected_position(
            db=db,
            position=position,
            market_exit_status="STRATEGY_MARKET_FILLED",
            market_exit_notes=f"Strategy-requested Spot exit for position #{position.id}",
        )

    async def _close_protected_position(
        self,
        *,
        db: AsyncSession,
        position: Position,
        market_exit_status: Literal["MANUAL_MARKET_FILLED", "STRATEGY_MARKET_FILLED"],
        market_exit_notes: str,
    ) -> ExchangePositionCloseReport:
        symbol = position.symbol
        protected_quantity = _decimal(position.protective_quantity)
        order_list_id = int(position.protective_order_list_id)

        position.protection_status = "EXITING"
        position.protection_updated_at = datetime.now(UTC)
        await db.commit()

        try:
            await self._exchange.cancel_spot_order_list(symbol, order_list_id)
            terminal_state = await self._read_terminal_oco(
                symbol, order_list_id, protected_quantity
            )
        except Exception as exc:
            await self._mark_missing(db, position, "could not cancel and verify native OCO")
            raise SpotPositionCloseError(
                "native OCO cancellation could not be verified; "
                "manual Binance intervention is required"
            ) from exc

        if terminal_state.kind == "SELL_FILLED":
            assert terminal_state.exit_order is not None
            return await self._record_oco_exit(db, position, terminal_state.exit_order)
        if terminal_state.kind != "CANCELLED":
            await self._mark_missing(db, position, "terminal OCO state was ambiguous")
            raise SpotPositionCloseError(
                "native OCO reached an ambiguous terminal state; "
                "manual Binance intervention is required"
            )

        try:
            submitted_market_exit = await self._exchange.place_market_order(
                symbol,
                "SELL",
                protected_quantity,
                client_order_id=f"TM-X-{_token()}",
            )
            market_exit = await self._read_market_exit(symbol, submitted_market_exit)
        except Exception as exc:
            await self._mark_missing(db, position, "Spot market exit submission failed")
            raise SpotPositionCloseError(
                "native OCO was cancelled but the exchange exit could not be submitted; "
                "manual Binance intervention is required"
            ) from exc

        return await self._record_market_exit(
            db,
            position,
            market_exit,
            protected_quantity,
            completed_status=market_exit_status,
            notes=market_exit_notes,
        )

    @staticmethod
    async def _load_protected_position(
        db: AsyncSession,
        position_id: int,
        *,
        execution_mode: TradingExecutionMode,
    ) -> Position:
        result = await db.execute(
            select(Position)
            .where(Position.id == position_id, Position.is_open.is_(True))
            .with_for_update()
        )
        position = result.scalar_one_or_none()
        if position is None:
            raise SpotPositionCloseError("open position was not found")
        if position.execution_mode != execution_mode.value:
            raise SpotPositionCloseError(
                f"exchange close requires a {execution_mode.value} position in the active ledger"
            )
        if position.side != "LONG":
            raise SpotPositionCloseError("Binance Spot exchange close only supports LONG positions")
        protected_quantity = _decimal(position.protective_quantity)
        if (
            position.protection_status != "ACTIVE"
            or position.protective_order_list_id is None
            or protected_quantity <= 0
            or protected_quantity != _decimal(position.quantity)
        ):
            raise SpotPositionCloseError(
                "position does not have a complete active native OCO; "
                "manual Binance intervention is required"
            )
        return position

    async def _read_terminal_oco(
        self,
        symbol: str,
        order_list_id: int,
        protected_quantity: Decimal,
    ) -> _TerminalOcoState:
        order_list = await self._exchange.get_spot_order_list(order_list_id)
        if (
            _order_list_id(order_list) != order_list_id
            or order_list.get("symbol") != symbol
            or order_list.get("contingencyType") != "OCO"
            or order_list.get("listStatusType") != "ALL_DONE"
            or order_list.get("listOrderStatus") != "ALL_DONE"
        ):
            return _TerminalOcoState("AMBIGUOUS")

        order_ids = _child_order_ids(order_list)
        if len(order_ids) != 2:
            return _TerminalOcoState("AMBIGUOUS")
        child_orders = [
            await self._exchange.get_order_status(symbol, order_id) for order_id in order_ids
        ]
        filled_sells = [
            order
            for order in child_orders
            if order.get("side") == "SELL" and _decimal(order.get("executedQty")) > 0
        ]
        if filled_sells:
            if len(filled_sells) != 1:
                return _TerminalOcoState("AMBIGUOUS")
            exit_order = filled_sells[0]
            if (
                exit_order.get("status") != "FILLED"
                or int(exit_order.get("orderListId", -1)) != order_list_id
                or _decimal(exit_order.get("executedQty")) != protected_quantity
                or _average_fill_price(exit_order) is None
            ):
                return _TerminalOcoState("AMBIGUOUS")
            return _TerminalOcoState("SELL_FILLED", exit_order)

        if any(
            order.get("side") != "SELL"
            or order.get("status") not in {"CANCELED", "EXPIRED"}
            or _decimal(order.get("executedQty")) != 0
            for order in child_orders
        ):
            return _TerminalOcoState("AMBIGUOUS")
        return _TerminalOcoState("CANCELLED")

    async def _read_market_exit(
        self, symbol: str, submitted_market_exit: dict[str, Any]
    ) -> dict[str, Any]:
        exchange_order_id = str(submitted_market_exit.get("orderId", ""))
        if not exchange_order_id:
            raise SpotPositionCloseError("exchange market exit did not return an order id")
        try:
            order_id = int(exchange_order_id)
        except ValueError as exc:
            raise SpotPositionCloseError(
                "exchange market exit returned an invalid order id"
            ) from exc
        verified_market_exit = await self._exchange.get_order_status(symbol, order_id)
        if (
            str(verified_market_exit.get("orderId", "")) != exchange_order_id
            or verified_market_exit.get("symbol") != symbol
            or verified_market_exit.get("side") != "SELL"
        ):
            raise SpotPositionCloseError(
                "exchange market exit signed read did not match submission"
            )
        return verified_market_exit

    async def _record_oco_exit(
        self,
        db: AsyncSession,
        position: Position,
        exit_order: dict[str, Any],
    ) -> ExchangePositionCloseReport:
        exit_price = _average_fill_price(exit_order)
        if exit_price is None:
            await self._mark_missing(db, position, "OCO fill had no usable average price")
            raise SpotPositionCloseError("native OCO fill did not include a usable average price")
        exit_order_id = str(exit_order["orderId"])
        await self._record_exit_order(
            db,
            position=position,
            exchange_order_id=exit_order_id,
            status=OrderStatus.FILLED,
            filled_quantity=_decimal(exit_order.get("executedQty")),
            exit_price=exit_price,
            notes=f"Native OCO exit for position #{position.id}",
        )
        await portfolio_tracker.close_position(db, position, float(exit_price))
        position.protection_status = "EXIT_FILLED"
        position.protection_updated_at = datetime.now(UTC)
        await db.commit()
        return ExchangePositionCloseReport(
            status="OCO_FILLED",
            position_id=position.id,
            symbol=position.symbol,
            exit_order_id=exit_order_id,
            exit_price=exit_price,
            closed_at=position.closed_at or datetime.now(UTC),
        )

    async def _record_market_exit(
        self,
        db: AsyncSession,
        position: Position,
        market_exit: dict[str, Any],
        requested_quantity: Decimal,
        *,
        completed_status: Literal["MANUAL_MARKET_FILLED", "STRATEGY_MARKET_FILLED"],
        notes: str,
    ) -> ExchangePositionCloseReport:
        exit_order_id = str(market_exit.get("orderId", ""))
        filled_quantity = _decimal(market_exit.get("executedQty"))
        exit_price = _average_fill_price(market_exit)
        if not exit_order_id or filled_quantity <= 0 or exit_price is None:
            await self._mark_missing(db, position, "market exit response was incomplete")
            raise SpotPositionCloseError(
                "exchange market exit response was incomplete; "
                "manual Binance intervention is required"
            )

        raw_status = str(market_exit.get("status", ""))
        status = _order_status(raw_status)
        await self._record_exit_order(
            db,
            position=position,
            exchange_order_id=exit_order_id,
            status=status,
            filled_quantity=filled_quantity,
            exit_price=exit_price,
            notes=notes,
        )

        if status == OrderStatus.FILLED and filled_quantity == requested_quantity:
            await portfolio_tracker.close_position(db, position, float(exit_price))
            position.protection_status = "EXIT_FILLED"
            position.protection_updated_at = datetime.now(UTC)
            await db.commit()
            return ExchangePositionCloseReport(
                status=completed_status,
                position_id=position.id,
                symbol=position.symbol,
                exit_order_id=exit_order_id,
                exit_price=exit_price,
                closed_at=position.closed_at or datetime.now(UTC),
            )

        actual_fill = min(requested_quantity, filled_quantity)
        remaining_quantity = requested_quantity - actual_fill
        realized_pnl = (exit_price - _decimal(position.entry_price)) * actual_fill
        position.quantity = float(remaining_quantity)
        position.current_price = float(exit_price)
        position.realized_pnl = float(_decimal(position.realized_pnl) + realized_pnl)
        position.unrealized_pnl = 0
        position.protective_order_list_id = None
        position.protective_quantity = None
        await self._mark_missing(db, position, "market exit was partial")
        raise SpotPositionCloseError(
            "market exit was only partially filled; the remaining position is marked MISSING"
        )

    async def _record_exit_order(
        self,
        db: AsyncSession,
        *,
        position: Position,
        exchange_order_id: str,
        status: OrderStatus,
        filled_quantity: Decimal,
        exit_price: Decimal,
        notes: str,
    ) -> Order:
        order = Order(
            exchange_order_id=exchange_order_id,
            symbol=position.symbol,
            side="SELL",
            order_type=OrderType.MARKET,
            status=status,
            quantity=float(position.quantity),
            price=float(exit_price),
            filled_quantity=float(filled_quantity),
            avg_fill_price=float(exit_price),
            commission=0,
            execution_mode=position.execution_mode,
            notes=notes,
        )
        db.add(order)
        await db.flush()
        return order

    async def _mark_missing(self, db: AsyncSession, position: Position, reason: str) -> None:
        position.protection_status = "MISSING"
        position.protection_updated_at = datetime.now(UTC)
        if position.execution_mode == TradingExecutionMode.LIVE.value:
            live_trading_guard.disarm(reason)
        await db.commit()
        logger.critical(
            "live_spot_position_close_unresolved",
            position_id=position.id,
            symbol=position.symbol,
            reason=reason,
        )


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


def _average_fill_price(order: dict[str, Any]) -> Decimal | None:
    quantity = _decimal(order.get("executedQty"))
    quote = _decimal(order.get("cummulativeQuoteQty"))
    if quantity > 0 and quote > 0:
        return quote / quantity
    price = _decimal(order.get("price"))
    return price if price > 0 else None


def _order_status(value: str) -> OrderStatus:
    try:
        return OrderStatus(value)
    except ValueError:
        return OrderStatus.SUBMITTED


def _decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value or 0))
    except Exception:
        return Decimal("0")


def _token() -> str:
    import uuid

    return uuid.uuid4().hex[:20]


spot_position_closer = SpotPositionCloser()
