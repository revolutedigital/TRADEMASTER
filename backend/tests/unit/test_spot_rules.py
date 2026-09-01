"""Tests for Binance Spot exchange-filter validation."""

from decimal import Decimal

import pytest

from app.services.exchange.spot_rules import SpotRuleViolation, SpotSymbolRules


def _exchange_info() -> dict:
    return {
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
            {
                "filterType": "NOTIONAL",
                "minNotional": "10.00",
                "maxNotional": "1000000.00",
            },
        ],
    }


def test_spot_rules_align_quantity_and_validate_long_exit_oco() -> None:
    rules = SpotSymbolRules.from_exchange_info(_exchange_info())

    quantity, take_profit, stop_loss = rules.prepare_long_exit_oco(
        last_price=Decimal("90000.05"),
        quantity=Decimal("0.00129"),
        take_profit_price=Decimal("92000.09"),
        stop_loss_price=Decimal("88000.09"),
    )

    assert quantity == Decimal("0.00120")
    assert take_profit == Decimal("92000.00")
    assert stop_loss == Decimal("88000.00")


def test_spot_rules_reject_an_oco_with_invalid_price_ordering() -> None:
    rules = SpotSymbolRules.from_exchange_info(_exchange_info())

    with pytest.raises(SpotRuleViolation, match="requires take profit above"):
        rules.prepare_long_exit_oco(
            last_price=Decimal("90000"),
            quantity=Decimal("0.001"),
            take_profit_price=Decimal("89000"),
            stop_loss_price=Decimal("88000"),
        )


def test_oco_quantity_must_satisfy_market_and_limit_lot_filters() -> None:
    exchange_info = _exchange_info()
    exchange_info["filters"].append(
        {
            "filterType": "LOT_SIZE",
            "minQty": "0.005",
            "maxQty": "100.000",
            "stepSize": "0.005",
        }
    )
    rules = SpotSymbolRules.from_exchange_info(exchange_info)

    assert rules.normalize_market_quantity(Decimal("0.012")) == Decimal("0.0120")
    assert rules.normalize_oco_quantity(Decimal("0.012")) == Decimal("0.010")


def test_minimum_market_quantity_rounds_up_to_meet_notional_filter() -> None:
    rules = SpotSymbolRules.from_exchange_info(_exchange_info())

    quantity = rules.minimum_market_quantity_for_notional(Decimal("90000"))

    assert quantity == Decimal("0.00020")
