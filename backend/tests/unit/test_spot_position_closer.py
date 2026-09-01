"""Tests for the controlled manual LIVE Spot close workflow."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import TradingExecutionMode
from app.models.portfolio import Position
from app.models.trade import Order
from app.services.exchange.spot_position_closer import (
    SpotPositionCloseError,
    SpotPositionCloser,
)


def _position(*, execution_mode: str = "LIVE") -> Position:
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
        execution_mode=execution_mode,
        protective_order_list_id=44,
        protective_quantity=0.001,
        protection_status="ACTIVE",
    )


def _database_for(position: Position) -> AsyncMock:
    database = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = position
    database.execute = AsyncMock(return_value=result)
    database.add = MagicMock()
    database.flush = AsyncMock()
    database.commit = AsyncMock()
    return database


class _Exchange:
    def __init__(self, *, child_orders: dict[int, dict], market_exit: dict | None = None) -> None:
        self._child_orders = child_orders
        self._market_exit = market_exit
        self.market_exit_calls = 0

    async def cancel_spot_order_list(self, symbol: str, order_list_id: int) -> dict:
        assert symbol == "BTCUSDT"
        assert order_list_id == 44
        return {"orderListId": 44}

    async def get_spot_order_list(self, order_list_id: int) -> dict:
        assert order_list_id == 44
        return {
            "orderListId": 44,
            "symbol": "BTCUSDT",
            "contingencyType": "OCO",
            "listStatusType": "ALL_DONE",
            "listOrderStatus": "ALL_DONE",
            "orders": [{"orderId": 91}, {"orderId": 92}],
        }

    async def get_order_status(self, symbol: str, order_id: int) -> dict:
        assert symbol == "BTCUSDT"
        if order_id == 103:
            assert self._market_exit is not None
            return {
                **self._market_exit,
                "symbol": "BTCUSDT",
                "side": "SELL",
            }
        return self._child_orders[order_id]

    async def place_market_order(
        self, symbol: str, side: str, quantity: Decimal, client_order_id: str
    ) -> dict:
        assert symbol == "BTCUSDT"
        assert side == "SELL"
        assert quantity == Decimal("0.001")
        assert client_order_id.startswith("TM-X-")
        self.market_exit_calls += 1
        assert self._market_exit is not None
        return self._market_exit


def _cancelled_children() -> dict[int, dict]:
    return {
        91: {"orderListId": 44, "side": "SELL", "status": "CANCELED", "executedQty": "0"},
        92: {"orderListId": 44, "side": "SELL", "status": "EXPIRED", "executedQty": "0"},
    }


@pytest.mark.asyncio
async def test_manual_close_cancels_oco_then_records_one_full_market_exit() -> None:
    position = _position()
    database = _database_for(position)
    exchange = _Exchange(
        child_orders=_cancelled_children(),
        market_exit={
            "orderId": 103,
            "status": "FILLED",
            "executedQty": "0.001",
            "cummulativeQuoteQty": "91.2",
        },
    )

    with (
        patch("app.services.exchange.spot_position_closer.live_trading_guard") as guard,
        patch("app.services.portfolio.tracker.event_bus") as event_bus,
    ):
        event_bus.publish = AsyncMock()
        report = await SpotPositionCloser(exchange).close(
            db=database,
            position_id=7,
            totp_code="123456",
        )

    assert report.status == "MANUAL_MARKET_FILLED"
    assert report.exit_order_id == "103"
    assert position.is_open is False
    assert position.protection_status == "EXIT_FILLED"
    assert exchange.market_exit_calls == 1
    assert any(isinstance(call.args[0], Order) for call in database.add.call_args_list)
    guard.require_live_exit.assert_called_once_with("123456")
    guard.disarm.assert_called_once_with("operator requested a live Spot position exit")


@pytest.mark.asyncio
async def test_manual_close_never_sends_second_sell_when_oco_already_filled() -> None:
    position = _position()
    database = _database_for(position)
    exchange = _Exchange(
        child_orders={
            91: {
                "orderId": 91,
                "orderListId": 44,
                "side": "SELL",
                "status": "FILLED",
                "executedQty": "0.001",
                "cummulativeQuoteQty": "91.2",
            },
            92: {"orderListId": 44, "side": "SELL", "status": "CANCELED", "executedQty": "0"},
        }
    )

    with (
        patch("app.services.exchange.spot_position_closer.live_trading_guard"),
        patch("app.services.portfolio.tracker.event_bus") as event_bus,
    ):
        event_bus.publish = AsyncMock()
        report = await SpotPositionCloser(exchange).close(
            db=database,
            position_id=7,
            totp_code="123456",
        )

    assert report.status == "OCO_FILLED"
    assert position.is_open is False
    assert exchange.market_exit_calls == 0


@pytest.mark.asyncio
async def test_manual_close_marks_remaining_position_missing_after_partial_market_exit() -> None:
    position = _position()
    database = _database_for(position)
    exchange = _Exchange(
        child_orders=_cancelled_children(),
        market_exit={
            "orderId": 103,
            "status": "PARTIALLY_FILLED",
            "executedQty": "0.0004",
            "cummulativeQuoteQty": "36.4",
        },
    )

    with (
        patch("app.services.exchange.spot_position_closer.live_trading_guard") as guard,
        pytest.raises(SpotPositionCloseError, match="partially filled"),
    ):
        await SpotPositionCloser(exchange).close(
            db=database,
            position_id=7,
            totp_code="123456",
        )

    assert position.is_open is True
    assert position.protection_status == "MISSING"
    assert float(position.quantity) == pytest.approx(0.0006)
    assert guard.disarm.call_args_list[-1].args == ("market exit was partial",)


@pytest.mark.asyncio
async def test_strategy_exit_on_testnet_cancels_oco_before_selling_the_tracked_long() -> None:
    position = _position(execution_mode="TESTNET")
    database = _database_for(position)
    exchange = _Exchange(
        child_orders=_cancelled_children(),
        market_exit={
            "orderId": 103,
            "status": "FILLED",
            "executedQty": "0.001",
            "cummulativeQuoteQty": "91.2",
        },
    )

    with (
        patch("app.services.exchange.spot_position_closer.live_trading_guard") as guard,
        patch("app.services.portfolio.tracker.event_bus") as event_bus,
    ):
        event_bus.publish = AsyncMock()
        report = await SpotPositionCloser(exchange).close_for_strategy(
            db=database,
            position_id=7,
            execution_mode=TradingExecutionMode.TESTNET,
        )

    assert report.status == "STRATEGY_MARKET_FILLED"
    assert position.is_open is False
    assert position.protection_status == "EXIT_FILLED"
    assert exchange.market_exit_calls == 1
    assert guard.require_live_strategy_exit.call_count == 0
    recorded_order = next(
        call.args[0] for call in database.add.call_args_list if isinstance(call.args[0], Order)
    )
    assert recorded_order.execution_mode == "TESTNET"
