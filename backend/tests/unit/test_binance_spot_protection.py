"""Tests for exchange-native Binance Spot protection requests."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.exchange.binance_client import BinanceClientWrapper, NativeSpotOcoProtection
from app.services.exchange.spot_rules import SpotSymbolRules


def _rules() -> SpotSymbolRules:
    return SpotSymbolRules.from_exchange_info(
        {
            "symbol": "BTCUSDT",
            "status": "TRADING",
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                {
                    "filterType": "MARKET_LOT_SIZE",
                    "minQty": "0.00010",
                    "maxQty": "100.00000",
                    "stepSize": "0.00010",
                },
                {"filterType": "NOTIONAL", "minNotional": "10.00"},
            ],
        }
    )


@pytest.mark.asyncio
async def test_native_oco_uses_current_rules_and_decimal_wire_values() -> None:
    wrapper = BinanceClientWrapper()
    exchange_client = MagicMock()
    exchange_client.v3_post_order_list_oco = AsyncMock(return_value={"orderListId": 42})
    wrapper._client = exchange_client
    wrapper.get_spot_symbol_rules = AsyncMock(return_value=_rules())

    result = await wrapper.place_spot_long_exit_oco(
        symbol="BTCUSDT",
        last_price=Decimal("90000.05"),
        quantity=Decimal("0.00129"),
        take_profit_price=Decimal("92000.09"),
        stop_loss_price=Decimal("88000.09"),
        client_order_id="TM-PROTECT-42",
    )

    assert result == NativeSpotOcoProtection(
        order_list_id=42,
        protected_quantity=Decimal("0.00120"),
        response={"orderListId": 42},
    )
    exchange_client.v3_post_order_list_oco.assert_awaited_once_with(
        symbol="BTCUSDT",
        side="SELL",
        quantity="0.0012",
        aboveType="LIMIT_MAKER",
        abovePrice="92000.00",
        belowType="STOP_LOSS",
        belowStopPrice="88000.00",
        listClientOrderId="TM-PROTECT-42",
    )


@pytest.mark.asyncio
async def test_open_oco_reconciliation_query_has_no_unsupported_symbol_filter() -> None:
    wrapper = BinanceClientWrapper()
    exchange_client = MagicMock()
    exchange_client.v3_get_open_order_list = AsyncMock(return_value=[])
    wrapper._client = exchange_client

    result = await wrapper.get_open_spot_order_lists()

    assert result == []
    exchange_client.v3_get_open_order_list.assert_awaited_once_with()
