"""Tests for the forex instrument model: pips, lots, conversion, sizing and P&L."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from app.fx.instruments import (
    STANDARD_LOT_UNITS,
    ConversionRates,
    Instrument,
    MissingRateError,
    UnknownInstrumentError,
    floor_to_lot_step,
    lots_from_units,
    margin_required,
    max_stop_pips_for_min_lot,
    notional,
    pip_value,
    profit,
    size_for_risk,
    units_from_lots,
)

RATES = ConversionRates(
    {
        "EURUSD": 1.10, "GBPUSD": 1.27, "AUDUSD": 0.66, "NZDUSD": 0.60,
        "USDJPY": 150.0, "USDCAD": 1.35, "USDCHF": 0.90, "USDBRL": 5.50,
    }
)
EURUSD = Instrument.from_symbol("EURUSD")
USDJPY = Instrument.from_symbol("USDJPY")
EURJPY = Instrument.from_symbol("EURJPY")
EURGBP = Instrument.from_symbol("EURGBP")


def test_instruments_derive_pip_size_and_decimals_from_the_quote_currency() -> None:
    assert (EURUSD.pip_size, EURUSD.price_decimals) == (0.0001, 5)
    assert (USDJPY.pip_size, USDJPY.price_decimals) == (0.01, 3)
    assert (EURJPY.base, EURJPY.quote) == ("EUR", "JPY")
    assert EURUSD.min_units == 1_000


@pytest.mark.parametrize("symbol", ["EUR/USD", "eurusd", "EURUS", "EURUSDX", ""])
def test_malformed_symbols_are_rejected(symbol: str) -> None:
    with pytest.raises(UnknownInstrumentError):
        Instrument.from_symbol(symbol)


def test_usd_per_unit_uses_direct_and_inverse_quotes() -> None:
    assert RATES.usd_per("USD") == 1.0
    assert RATES.usd_per("EUR") == pytest.approx(1.10)
    assert RATES.usd_per("JPY") == pytest.approx(1 / 150.0)
    assert RATES.usd_per("BRL") == pytest.approx(1 / 5.5)


def test_a_missing_or_invalid_rate_fails_closed() -> None:
    with pytest.raises(MissingRateError):
        RATES.usd_per("SEK")
    with pytest.raises(ValueError):
        ConversionRates({"EURUSD": 0.0}).usd_per("EUR")
    with pytest.raises(ValueError):
        ConversionRates({"USDJPY": float("nan")}).usd_per("JPY")


def test_pip_values_of_one_standard_lot_in_usd() -> None:
    lot = STANDARD_LOT_UNITS

    assert pip_value(EURUSD, lot, RATES) == pytest.approx(10.0)
    assert pip_value(USDJPY, lot, RATES) == pytest.approx(1000 / 150.0)
    assert pip_value(EURJPY, lot, RATES) == pytest.approx(1000 / 150.0)  # JPY quote, valued at USDJPY
    assert pip_value(EURGBP, lot, RATES) == pytest.approx(10 * 1.27)  # GBP quote, valued at GBPUSD


def test_pip_value_in_another_account_currency() -> None:
    reais = pip_value(EURUSD, STANDARD_LOT_UNITS, RATES, account_currency="BRL")

    assert reais == pytest.approx(10.0 * 5.5)
    with pytest.raises(ValueError):
        pip_value(EURUSD, 0, RATES)


def test_lot_conversions_and_flooring_never_round_up() -> None:
    assert units_from_lots(0.01) == 1_000
    assert lots_from_units(250_000) == pytest.approx(2.5)
    assert floor_to_lot_step(1_999, EURUSD) == 1_000
    assert floor_to_lot_step(999, EURUSD) == 0
    assert floor_to_lot_step(-5, EURUSD) == 0


def test_floor_to_lot_step_tolerates_float_boundaries_without_under_sizing() -> None:
    assert floor_to_lot_step(11_999.999999999998, EURUSD) == 12_000
    assert floor_to_lot_step(11_999.99, EURUSD) == 11_000


def test_sizing_takes_the_largest_lot_inside_the_risk_budget() -> None:
    # $5,000 at 1% risk is $50; a 25 pip stop risks $2.50 per 0.10 lot? Check with per-pip value.
    result = size_for_risk(
        EURUSD, equity=5_000, risk_fraction=0.01, stop_distance_pips=25, rates=RATES
    )

    assert result.tradable
    assert result.units == 20_000  # 0.20 lot: $2/pip x 25 pips = $50
    assert result.risk_at_stop == pytest.approx(50.0)
    assert result.risk_at_stop <= result.risk_budget


def test_a_small_account_cannot_take_a_wide_stop_with_the_minimum_lot() -> None:
    """The canary: $500 at 0.25% risk is $1.25, and the smallest lot is $0.10 per pip."""
    tight = size_for_risk(
        EURUSD, equity=500, risk_fraction=0.0025, stop_distance_pips=12, rates=RATES
    )
    wide = size_for_risk(
        EURUSD, equity=500, risk_fraction=0.0025, stop_distance_pips=20, rates=RATES
    )

    assert tight.units == 1_000 and tight.risk_at_stop == pytest.approx(1.20)
    assert not wide.tradable and "above" in wide.reason
    assert max_stop_pips_for_min_lot(
        EURUSD, equity=500, risk_fraction=0.0025, rates=RATES
    ) == pytest.approx(12.5)


def test_the_yen_pair_allows_a_wider_stop_because_its_pip_is_worth_less() -> None:
    assert max_stop_pips_for_min_lot(
        USDJPY, equity=500, risk_fraction=0.0025, rates=RATES
    ) == pytest.approx(1.25 / (1000 / 150.0 / 100))  # about 18.75 pips


def test_sizing_rejects_nonsense_inputs() -> None:
    for bad in (
        dict(equity=0, risk_fraction=0.01, stop_distance_pips=10),
        dict(equity=500, risk_fraction=0.0, stop_distance_pips=10),
        dict(equity=500, risk_fraction=1.5, stop_distance_pips=10),
        dict(equity=500, risk_fraction=0.01, stop_distance_pips=0),
    ):
        with pytest.raises(ValueError):
            size_for_risk(EURUSD, rates=RATES, **bad)


def test_profit_is_signed_by_side_and_valued_at_the_exit_rate() -> None:
    long_win = profit(EURUSD, side="LONG", units=100_000, entry_price=1.10, exit_price=1.105, rates=RATES)
    short_win = profit(EURUSD, side="SHORT", units=100_000, entry_price=1.105, exit_price=1.10, rates=RATES)
    jpy = profit(USDJPY, side="LONG", units=100_000, entry_price=150.0, exit_price=151.0, rates=RATES)
    cross = profit(EURJPY, side="LONG", units=100_000, entry_price=160.0, exit_price=161.0, rates=RATES)

    assert long_win == pytest.approx(500.0) and short_win == pytest.approx(500.0)
    assert jpy == pytest.approx(100_000 * 1.0 / 150.0)
    assert cross == pytest.approx(100_000 * 1.0 / 150.0)
    with pytest.raises(ValueError):
        profit(EURUSD, side="LONG", units=0, entry_price=1.1, exit_price=1.1, rates=RATES)


def test_notional_and_margin_scale_with_leverage() -> None:
    assert notional(EURUSD, units=100_000, price=1.10, rates=RATES) == pytest.approx(110_000)
    assert margin_required(
        EURUSD, units=100_000, price=1.10, leverage=100, rates=RATES
    ) == pytest.approx(1_100)
    with pytest.raises(ValueError):
        margin_required(EURUSD, units=1, price=1.1, leverage=0.5, rates=RATES)


@given(
    equity=st.floats(min_value=100, max_value=1_000_000),
    risk=st.floats(min_value=0.0005, max_value=0.05),
    stop=st.floats(min_value=1, max_value=500),
    symbol=st.sampled_from(["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "EURJPY", "EURGBP"]),
)
def test_a_sized_position_never_loses_more_than_the_budget_at_its_stop(equity, risk, stop, symbol) -> None:
    instrument = Instrument.from_symbol(symbol)

    result = size_for_risk(
        instrument, equity=equity, risk_fraction=risk, stop_distance_pips=stop, rates=RATES
    )

    if result.tradable:
        assert result.units % 1_000 == 0
        assert result.risk_at_stop <= result.risk_budget * (1 + 1e-9)
        # One more step would have broken the budget.
        one_more = (result.units + 1_000) * (result.risk_at_stop / result.units)
        assert one_more > result.risk_budget
    else:
        smallest_lot_risk = pip_value(instrument, instrument.min_units, RATES) * stop
        assert smallest_lot_risk > result.risk_budget


@given(units=st.floats(min_value=1, max_value=10_000_000), symbol=st.sampled_from(["EURUSD", "USDJPY", "EURGBP"]))
def test_pip_value_is_linear_in_the_position_size(units, symbol) -> None:
    instrument = Instrument.from_symbol(symbol)

    single = pip_value(instrument, units, RATES)
    double = pip_value(instrument, 2 * units, RATES)

    assert double == pytest.approx(2 * single)


@given(
    amount=st.floats(min_value=0.01, max_value=1e9),
    source=st.sampled_from(["USD", "EUR", "GBP", "JPY", "CAD", "CHF", "AUD", "NZD", "BRL"]),
    target=st.sampled_from(["USD", "EUR", "GBP", "JPY", "CAD", "CHF", "AUD", "NZD", "BRL"]),
)
def test_converting_there_and_back_returns_the_original_amount(amount, source, target) -> None:
    there = RATES.convert(amount, source=source, target=target)
    back = RATES.convert(there, source=target, target=source)

    assert back == pytest.approx(amount, rel=1e-9)


@given(a=st.floats(min_value=1, max_value=1e6), b=st.floats(min_value=1, max_value=1e6))
def test_conversion_through_the_dollar_is_consistent_for_cross_rates(a, b) -> None:
    # EUR -> JPY directly must equal EUR -> USD -> JPY: the definition of a synthetic cross.
    eurjpy = RATES.usd_per("EUR") / RATES.usd_per("JPY")

    assert RATES.convert(a, source="EUR", target="JPY") == pytest.approx(a * eurjpy)
    assert eurjpy == pytest.approx(1.10 * 150.0)
