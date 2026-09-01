"""Tests for fail-closed reconciliation of native Binance Spot OCO exits."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.portfolio import Position
from app.services.exchange.live_execution_readiness import live_protection_readiness
from app.services.exchange.spot_protection_reconciler import SpotProtectionReconciler


def _position() -> Position:
    return Position(
        id=7,
        symbol="BTCUSDT",
        side="LONG",
        entry_price=90_000,
        quantity=0.001,
        current_price=90_000,
        unrealized_pnl=0,
        realized_pnl=0,
        is_open=True,
        execution_mode="LIVE",
        protective_order_list_id=44,
        protective_quantity=0.001,
        protection_status="ACTIVE",
    )


def _database_for(position: Position) -> AsyncMock:
    database = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = [position]
    database.execute = AsyncMock(return_value=result)
    database.flush = AsyncMock()
    return database


class _Exchange:
    def __init__(
        self,
        *,
        open_lists: list[dict],
        order_list: dict | None = None,
        orders: dict[int, dict] | None = None,
        account: dict | None = None,
        open_orders: list[dict] | None = None,
    ):
        self._open_lists = open_lists
        self._order_list = order_list
        self._orders = orders or {}
        self._account = account or {"canTrade": True, "balances": []}
        self._open_orders = open_orders or []

    async def get_open_spot_order_lists(self) -> list[dict]:
        return self._open_lists

    async def get_spot_order_list(self, order_list_id: int) -> dict:
        assert order_list_id == 44
        if self._order_list is None:
            raise RuntimeError("order list not found")
        return self._order_list

    async def get_order_status(self, symbol: str, order_id: int) -> dict:
        assert symbol == "BTCUSDT"
        return self._orders[order_id]

    async def get_account(self) -> dict:
        return self._account

    async def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        assert symbol is None
        return self._open_orders


@pytest.fixture(autouse=True)
def reset_protection_readiness():
    live_protection_readiness.reset()
    yield
    live_protection_readiness.reset()


@pytest.mark.asyncio
async def test_reconciler_confirms_matching_active_oco() -> None:
    position = _position()
    database = _database_for(position)
    exchange = _Exchange(
        open_lists=[
            {
                "orderListId": 44,
                "symbol": "BTCUSDT",
                "contingencyType": "OCO",
                "listStatusType": "EXEC_STARTED",
                "listOrderStatus": "EXECUTING",
            }
        ],
        account={
            "canTrade": True,
            "balances": [{"asset": "BTC", "free": "0", "locked": "0.001"}],
        },
        open_orders=[{"orderId": 91, "orderListId": 44}],
    )

    report = await SpotProtectionReconciler(exchange).reconcile(database)

    assert report.ready is True
    assert report.active_protections == 1
    assert position.protection_status == "ACTIVE"
    assert live_protection_readiness.is_ready(45) is True


@pytest.mark.asyncio
async def test_reconciler_closes_position_only_after_full_confirmed_oco_fill() -> None:
    position = _position()
    database = _database_for(position)
    exchange = _Exchange(
        open_lists=[],
        order_list={
            "orderListId": 44,
            "contingencyType": "OCO",
            "listStatusType": "ALL_DONE",
            "listOrderStatus": "ALL_DONE",
            "orders": [{"orderId": 91}, {"orderId": 92}],
        },
        orders={
            91: {
                "orderListId": 44,
                "side": "SELL",
                "status": "FILLED",
                "executedQty": "0.001",
                "cummulativeQuoteQty": "91.25",
            },
            92: {"orderListId": 44, "side": "SELL", "status": "CANCELED", "executedQty": "0"},
        },
        account={"canTrade": True, "balances": []},
    )

    with patch("app.services.portfolio.tracker.event_bus") as event_bus:
        event_bus.publish = AsyncMock()
        report = await SpotProtectionReconciler(exchange).reconcile(database)

    assert report.ready is True
    assert report.closed_positions == 1
    assert position.is_open is False
    assert position.protection_status == "EXIT_FILLED"
    assert position.realized_pnl == pytest.approx(1.25)


@pytest.mark.asyncio
async def test_reconciler_marks_unfilled_or_missing_oco_as_unsafe() -> None:
    position = _position()
    database = _database_for(position)
    exchange = _Exchange(
        open_lists=[],
        order_list={
            "orderListId": 44,
            "contingencyType": "OCO",
            "listStatusType": "ALL_DONE",
            "listOrderStatus": "ALL_DONE",
            "orders": [{"orderId": 91}, {"orderId": 92}],
        },
        orders={
            91: {"orderListId": 44, "side": "SELL", "status": "CANCELED", "executedQty": "0"},
            92: {"orderListId": 44, "side": "SELL", "status": "CANCELED", "executedQty": "0"},
        },
        account={
            "canTrade": True,
            "balances": [{"asset": "BTC", "free": "0", "locked": "0.001"}],
        },
    )

    report = await SpotProtectionReconciler(exchange).reconcile(database)

    assert report.ready is False
    assert report.issues
    assert position.is_open is True
    assert position.protection_status == "MISSING"
    assert live_protection_readiness.is_ready(45) is False
