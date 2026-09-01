"""Read-only inventory guard for a dedicated Binance Spot LIVE account."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

from app.config import settings
from app.models.portfolio import Position

ACCOUNT_BALANCE_TOLERANCE = Decimal("0.00000001")


class SpotAccountInventoryExchange(Protocol):
    async def get_account(self) -> dict[str, Any]: ...

    async def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class SpotAccountInventoryReport:
    """Read-only account state comparison against the local LIVE ledger."""

    issues: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.issues


class SpotAccountInventoryReconciler:
    """Reject untracked Spot inventory and orders before a new LIVE entry."""

    def __init__(self, exchange: SpotAccountInventoryExchange) -> None:
        self._exchange = exchange

    async def reconcile(
        self,
        positions: list[Position],
        *,
        execution_mode: str = "LIVE",
    ) -> SpotAccountInventoryReport:
        account = await self._exchange.get_account()
        open_orders = await self._exchange.get_open_orders()
        issues: list[str] = []

        if not account.get("canTrade", False):
            issues.append("Binance account cannot trade")

        exchange_positions = [
            position
            for position in positions
            if position.is_open and position.execution_mode == execution_mode
        ]
        expected_balances, balance_issues = _expected_base_balances(
            exchange_positions,
            execution_mode=execution_mode,
        )
        issues.extend(balance_issues)
        actual_balances = _account_balances(account)
        for asset, expected_quantity in expected_balances.items():
            actual_quantity = actual_balances.get(asset, Decimal("0"))
            if abs(actual_quantity - expected_quantity) > ACCOUNT_BALANCE_TOLERANCE:
                issues.append(
                    f"Binance {asset} balance {actual_quantity} does not match "
                    f"tracked {execution_mode} quantity {expected_quantity}"
                )

        allowed_assets = set(settings.live_trading_allowed_assets_list)
        for asset, actual_quantity in actual_balances.items():
            if (
                asset not in expected_balances
                and asset not in allowed_assets
                and actual_quantity > ACCOUNT_BALANCE_TOLERANCE
            ):
                issues.append(f"untracked Binance asset balance {asset}={actual_quantity}")

        expected_order_list_ids = {
            int(position.protective_order_list_id)
            for position in exchange_positions
            if position.protection_status == "ACTIVE"
            and position.protective_order_list_id is not None
        }
        for order in open_orders:
            order_list_id = _order_list_id(order)
            if order_list_id is None or order_list_id not in expected_order_list_ids:
                issues.append(
                    f"untracked open Binance Spot order {order.get('orderId', 'unknown')}"
                )

        return SpotAccountInventoryReport(issues=tuple(issues))


def _expected_base_balances(
    positions: list[Position],
    *,
    execution_mode: str,
) -> tuple[dict[str, Decimal], list[str]]:
    expected: dict[str, Decimal] = {}
    issues: list[str] = []
    for position in positions:
        asset = _base_asset(position.symbol)
        quantity = _decimal(position.quantity)
        if asset is None:
            issues.append(
                "cannot reconcile Binance account inventory for unsupported "
                f"symbol {position.symbol}"
            )
            continue
        if quantity <= 0:
            issues.append(
                f"tracked {execution_mode} position {position.id} has invalid quantity"
            )
            continue
        expected[asset] = expected.get(asset, Decimal("0")) + quantity
    return expected, issues


def _base_asset(symbol: str) -> str | None:
    normalized_symbol = symbol.upper()
    if normalized_symbol.endswith("USDT") and len(normalized_symbol) > 4:
        return normalized_symbol[:-4]
    return None


def _account_balances(account: dict[str, Any]) -> dict[str, Decimal]:
    balances: dict[str, Decimal] = {}
    for balance in account.get("balances", []):
        asset = str(balance.get("asset", "")).upper()
        if not asset:
            continue
        total = _decimal(balance.get("free")) + _decimal(balance.get("locked"))
        if total < 0:
            continue
        balances[asset] = total
    return balances


def _order_list_id(order: dict[str, Any]) -> int | None:
    try:
        order_list_id = int(order.get("orderListId", -1))
    except (TypeError, ValueError):
        return None
    return order_list_id if order_list_id >= 0 else None


def _decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value or 0))
    except Exception:
        return Decimal("0")
