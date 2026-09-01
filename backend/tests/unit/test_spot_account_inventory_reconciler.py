"""Tests for the read-only LIVE Binance account inventory guard."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.models.portfolio import Position
from app.services.exchange.spot_account_inventory_reconciler import (
    SpotAccountInventoryReconciler,
)


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


class _Exchange:
    def __init__(self, account: dict, open_orders: list[dict]) -> None:
        self._account = account
        self._open_orders = open_orders

    async def get_account(self) -> dict:
        return self._account

    async def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        assert symbol is None
        return self._open_orders


@pytest.mark.asyncio
async def test_inventory_accepts_only_tracked_base_balance_and_native_oco_orders() -> None:
    exchange = _Exchange(
        {"canTrade": True, "balances": [{"asset": "BTC", "free": "0", "locked": "0.001"}]},
        [{"orderId": 91, "orderListId": 44}],
    )
    settings = SimpleNamespace(live_trading_allowed_assets_list=["USDT", "BNB"])

    with patch("app.services.exchange.spot_account_inventory_reconciler.settings", settings):
        report = await SpotAccountInventoryReconciler(exchange).reconcile([_position()])

    assert report.ready is True


@pytest.mark.asyncio
async def test_inventory_rejects_untracked_asset_balance() -> None:
    exchange = _Exchange(
        {"canTrade": True, "balances": [{"asset": "DOGE", "free": "10", "locked": "0"}]},
        [],
    )
    settings = SimpleNamespace(live_trading_allowed_assets_list=["USDT", "BNB"])

    with patch("app.services.exchange.spot_account_inventory_reconciler.settings", settings):
        report = await SpotAccountInventoryReconciler(exchange).reconcile([])

    assert report.ready is False
    assert "untracked Binance asset balance DOGE=10" in report.issues


@pytest.mark.asyncio
async def test_inventory_rejects_any_open_order_outside_tracked_oco() -> None:
    exchange = _Exchange(
        {"canTrade": True, "balances": [{"asset": "BTC", "free": "0", "locked": "0.001"}]},
        [{"orderId": 101, "orderListId": -1}],
    )
    settings = SimpleNamespace(live_trading_allowed_assets_list=["USDT", "BNB"])

    with patch("app.services.exchange.spot_account_inventory_reconciler.settings", settings):
        report = await SpotAccountInventoryReconciler(exchange).reconcile([_position()])

    assert report.ready is False
    assert "untracked open Binance Spot order 101" in report.issues
