"""Tests for the FX cost primitives (spread, slippage, commission, pip value, P&L)."""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest
from hypothesis import given
from hypothesis import strategies as st

from app.services.backtest.fx_costs import (
    STANDARD_LOT_UNITS,
    FxCostModel,
    NoCommission,
    NotionalCommission,
    PerLotCommission,
    SwapSchedule,
    UnsupportedPairError,
    notional_usd,
    pip_size,
    pip_value_usd,
    rollover_days_charged,
    round_trip_cost,
    swap_pnl_usd,
    trade_pnl_usd,
)

RAW_ECN = PerLotCommission(usd_per_lot_per_side=2.25)


def test_pip_size_is_a_hundredth_for_yen_quotes_and_a_ten_thousandth_otherwise() -> None:
    assert pip_size("USDJPY") == 0.01
    assert pip_size("EURJPY") == 0.01
    assert pip_size("EURUSD") == 0.0001


def test_pip_value_of_one_standard_lot() -> None:
    assert pip_value_usd("EURUSD", 1.10) == pytest.approx(10.0)
    assert pip_value_usd("USDJPY", 150.0) == pytest.approx(100_000 * 0.01 / 150.0)
    assert pip_value_usd("USDCAD", 1.35) == pytest.approx(100_000 * 0.0001 / 1.35)
    assert pip_value_usd("EURUSD", 1.10, units=STANDARD_LOT_UNITS / 2) == pytest.approx(5.0)


def test_cross_pairs_are_rejected_until_a_conversion_rate_exists() -> None:
    with pytest.raises(UnsupportedPairError):
        pip_value_usd("EURGBP", 0.86)
    with pytest.raises(UnsupportedPairError):
        notional_usd("EURGBP", 0.86, 100_000)


@pytest.mark.parametrize("symbol", ["EUR/USD", "eurusd", "EURUS", "EURUSDX"])
def test_malformed_symbols_are_rejected(symbol: str) -> None:
    with pytest.raises(UnsupportedPairError):
        pip_size(symbol)


def test_non_positive_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        pip_value_usd("EURUSD", 0.0)
    with pytest.raises(ValueError):
        pip_value_usd("EURUSD", 1.1, units=0)


def test_eurusd_one_pip_spread_with_raw_commission_costs_the_expected_usd_per_lot() -> None:
    model = FxCostModel(commission=RAW_ECN)

    cost = round_trip_cost(
        model,
        symbol="EURUSD",
        side="LONG",
        units=STANDARD_LOT_UNITS,
        entry_bid=1.10000,
        entry_ask=1.10010,
        exit_bid=1.10000,
        exit_ask=1.10010,
    )

    # Buy at the ask, sell at the bid: one pip of spread ($10) plus $2.25 on each side.
    assert cost.spread_and_slippage_pips == pytest.approx(1.0)
    assert cost.commission_usd == pytest.approx(4.50)
    assert cost.total_usd == pytest.approx(14.50)


def test_a_short_pays_the_same_spread_entering_at_the_bid_and_exiting_at_the_ask() -> None:
    model = FxCostModel(commission=NoCommission())

    short = round_trip_cost(
        model,
        symbol="EURUSD",
        side="SHORT",
        units=STANDARD_LOT_UNITS,
        entry_bid=1.10000,
        entry_ask=1.10010,
        exit_bid=1.10000,
        exit_ask=1.10010,
    )

    assert short.spread_and_slippage_pips == pytest.approx(1.0)
    assert short.total_usd == pytest.approx(10.0)


def test_slippage_is_charged_on_both_sides_against_the_trader() -> None:
    model = FxCostModel(commission=NoCommission(), slippage_pips=0.2)

    long_entry = model.fill_price(
        symbol="EURUSD", side="LONG", action="ENTRY", bid=1.1000, ask=1.1001
    )
    long_exit = model.fill_price(
        symbol="EURUSD", side="LONG", action="EXIT", bid=1.1000, ask=1.1001
    )
    short_entry = model.fill_price(
        symbol="EURUSD", side="SHORT", action="ENTRY", bid=1.1000, ask=1.1001
    )
    short_exit = model.fill_price(
        symbol="EURUSD", side="SHORT", action="EXIT", bid=1.1000, ask=1.1001
    )

    assert long_entry == pytest.approx(1.1001 + 0.00002)
    assert long_exit == pytest.approx(1.1000 - 0.00002)
    assert short_entry == pytest.approx(1.1000 - 0.00002)
    assert short_exit == pytest.approx(1.1001 + 0.00002)


def test_round_trip_with_slippage_costs_the_spread_plus_two_slips() -> None:
    model = FxCostModel(commission=NoCommission(), slippage_pips=0.2)

    cost = round_trip_cost(
        model,
        symbol="EURUSD",
        side="LONG",
        units=STANDARD_LOT_UNITS,
        entry_bid=1.10000,
        entry_ask=1.10010,
        exit_bid=1.10000,
        exit_ask=1.10010,
    )

    assert cost.spread_and_slippage_pips == pytest.approx(1.0 + 0.4)
    assert cost.total_usd == pytest.approx(14.0)


def test_spread_multiplier_widens_the_quote_symmetrically_around_the_mid() -> None:
    model = FxCostModel(commission=NoCommission(), spread_multiplier=3.0)

    ask_side = model.fill_price(symbol="EURUSD", side="LONG", action="ENTRY", bid=1.1000, ask=1.1001)
    bid_side = model.fill_price(symbol="EURUSD", side="LONG", action="EXIT", bid=1.1000, ask=1.1001)

    assert ask_side == pytest.approx(1.10005 + 0.00015)
    assert bid_side == pytest.approx(1.10005 - 0.00015)


def test_a_spread_multiplier_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="tighter"):
        FxCostModel(commission=NoCommission(), spread_multiplier=0.5)


def test_usdjpy_spread_is_converted_to_usd_at_the_exit_price() -> None:
    model = FxCostModel(commission=NoCommission())

    cost = round_trip_cost(
        model,
        symbol="USDJPY",
        side="LONG",
        units=STANDARD_LOT_UNITS,
        entry_bid=150.00,
        entry_ask=150.01,
        exit_bid=150.00,
        exit_ask=150.01,
    )

    assert cost.spread_and_slippage_pips == pytest.approx(1.0, rel=1e-3)
    assert cost.total_usd == pytest.approx(pip_value_usd("USDJPY", 150.005), rel=1e-3)


def test_notional_commission_applies_its_per_order_minimum() -> None:
    ibkr_like = NotionalCommission(basis_points=0.2, minimum_usd=2.0)

    one_lot = ibkr_like.per_side_usd(
        units=100_000, notional_usd=notional_usd("EURUSD", 1.10, 100_000)
    )
    half_lot = ibkr_like.per_side_usd(
        units=50_000, notional_usd=notional_usd("EURUSD", 1.10, 50_000)
    )

    assert one_lot == pytest.approx(2.2)  # 0.2 bp of $110,000 is above the minimum
    assert half_lot == pytest.approx(2.0)  # 0.2 bp of $55,000 is $1.10, so the minimum applies


def test_commissions_reject_negative_rates() -> None:
    with pytest.raises(ValueError):
        PerLotCommission(usd_per_lot_per_side=-1)
    with pytest.raises(ValueError):
        NotionalCommission(basis_points=-0.1, minimum_usd=2)


def test_crossed_or_invalid_quotes_are_refused() -> None:
    model = FxCostModel(commission=NoCommission())

    with pytest.raises(ValueError, match="crossed"):
        model.fill_price(symbol="EURUSD", side="LONG", action="ENTRY", bid=1.1001, ask=1.1000)
    with pytest.raises(ValueError):
        model.fill_price(symbol="EURUSD", side="LONG", action="ENTRY", bid=0.0, ask=1.1)


def test_gross_pnl_for_longs_shorts_and_usd_base_pairs() -> None:
    assert trade_pnl_usd(
        symbol="EURUSD", side="LONG", units=100_000, entry_price=1.1000, exit_price=1.1050
    ) == pytest.approx(500.0)
    assert trade_pnl_usd(
        symbol="EURUSD", side="SHORT", units=100_000, entry_price=1.1050, exit_price=1.1000
    ) == pytest.approx(500.0)
    assert trade_pnl_usd(
        symbol="USDJPY", side="LONG", units=100_000, entry_price=150.0, exit_price=151.0
    ) == pytest.approx(100_000 * 1.0 / 151.0)


def test_a_model_with_no_costs_charges_nothing_on_a_flat_quote() -> None:
    model = FxCostModel(commission=NoCommission())

    cost = round_trip_cost(
        model,
        symbol="EURUSD",
        side="LONG",
        units=STANDARD_LOT_UNITS,
        entry_bid=1.1000,
        entry_ask=1.1000,
        exit_bid=1.1000,
        exit_ask=1.1000,
    )

    assert cost.total_usd == pytest.approx(0.0, abs=1e-9)


quotes = st.builds(
    lambda mid, spread_pips: (mid - spread_pips * 0.00005, mid + spread_pips * 0.00005),
    mid=st.floats(min_value=0.6, max_value=2.0),
    spread_pips=st.floats(min_value=0.0, max_value=6.0),
)


@given(
    entry=quotes,
    exit_=quotes,
    side=st.sampled_from(["LONG", "SHORT"]),
    units=st.floats(min_value=1_000, max_value=1_000_000),
    slippage=st.floats(min_value=0.0, max_value=2.0),
    multiplier=st.floats(min_value=1.0, max_value=5.0),
)
def test_round_trip_never_costs_less_than_trading_at_the_mid(
    entry, exit_, side, units, slippage, multiplier
) -> None:
    model = FxCostModel(
        commission=RAW_ECN, slippage_pips=slippage, spread_multiplier=multiplier
    )

    cost = round_trip_cost(
        model,
        symbol="EURUSD",
        side=side,
        units=units,
        entry_bid=entry[0],
        entry_ask=entry[1],
        exit_bid=exit_[0],
        exit_ask=exit_[1],
    )

    assert cost.total_usd >= -1e-6
    assert cost.commission_usd == pytest.approx(2 * 2.25 * units / STANDARD_LOT_UNITS)


@given(
    slippage=st.floats(min_value=0.0, max_value=2.0),
    extra_slippage=st.floats(min_value=0.0, max_value=2.0),
)
def test_more_slippage_never_reduces_the_cost(slippage: float, extra_slippage: float) -> None:
    def cost_with(slip: float) -> float:
        return round_trip_cost(
            FxCostModel(commission=NoCommission(), slippage_pips=slip),
            symbol="EURUSD",
            side="LONG",
            units=STANDARD_LOT_UNITS,
            entry_bid=1.1000,
            entry_ask=1.1001,
            exit_bid=1.1000,
            exit_ask=1.1001,
        ).total_usd

    assert cost_with(slippage + extra_slippage) >= cost_with(slippage) - 1e-9


NY = ZoneInfo("America/New_York")


def _ny(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=NY)


# 2024-01-09 is a Tuesday and 2024-07-09 is also a Tuesday, so both weeks share a weekday layout.


def test_a_position_that_never_crosses_the_rollover_pays_no_swap() -> None:
    assert rollover_days_charged(_ny(2024, 1, 9, 9), _ny(2024, 1, 9, 16, 59)) == 0


def test_carrying_from_wednesday_to_thursday_pays_three_days() -> None:
    # Wednesday 10:00 to Thursday 10:00 crosses Wednesday's 17:00 rollover, which is triple.
    assert rollover_days_charged(_ny(2024, 1, 10, 10), _ny(2024, 1, 11, 10)) == 3


def test_ordinary_weekday_rollover_is_a_single_day() -> None:
    assert rollover_days_charged(_ny(2024, 1, 9, 10), _ny(2024, 1, 10, 10)) == 1  # Tue -> Wed
    assert rollover_days_charged(_ny(2024, 1, 11, 10), _ny(2024, 1, 12, 10)) == 1  # Thu -> Fri


def test_holding_over_the_weekend_pays_the_friday_rollover_once() -> None:
    # The weekend was already paid by Wednesday's triple charge.
    assert rollover_days_charged(_ny(2024, 1, 12, 10), _ny(2024, 1, 15, 10)) == 1


def test_a_full_week_pays_seven_days_of_financing() -> None:
    # Tue, Wed (triple), Thu, Fri, Mon rollovers = 1 + 3 + 1 + 1 + 1.
    assert rollover_days_charged(_ny(2024, 1, 9, 10), _ny(2024, 1, 16, 10)) == 7


def test_the_rollover_follows_new_york_time_across_daylight_saving() -> None:
    # 17:00 New York is 22:00 UTC in winter and 21:00 UTC in summer.
    winter_before = datetime(2024, 1, 9, 21, 30, tzinfo=UTC)
    winter_after = datetime(2024, 1, 9, 22, 30, tzinfo=UTC)
    summer_before = datetime(2024, 7, 9, 20, 30, tzinfo=UTC)
    summer_after = datetime(2024, 7, 9, 21, 30, tzinfo=UTC)

    assert rollover_days_charged(winter_before, winter_after) == 1
    assert rollover_days_charged(summer_before, summer_after) == 1
    assert rollover_days_charged(winter_after, datetime(2024, 1, 10, 21, 30, tzinfo=UTC)) == 0
    assert rollover_days_charged(summer_after, datetime(2024, 7, 10, 20, 30, tzinfo=UTC)) == 0


def test_a_broker_can_move_the_triple_charge_to_another_weekday() -> None:
    thursday_to_friday = (_ny(2024, 1, 11, 10), _ny(2024, 1, 12, 10))

    assert rollover_days_charged(*thursday_to_friday, triple_weekday=3) == 3
    assert rollover_days_charged(*thursday_to_friday) == 1


def test_swap_rejects_naive_timestamps_and_reversed_time() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        rollover_days_charged(datetime(2024, 1, 9, 10), datetime(2024, 1, 10, 10))
    with pytest.raises(ValueError, match="before"):
        rollover_days_charged(_ny(2024, 1, 10, 10), _ny(2024, 1, 9, 10))
    with pytest.raises(ValueError, match="weekday"):
        SwapSchedule(long_pips_per_day=0, short_pips_per_day=0, triple_weekday=5)


def test_swap_is_signed_by_direction_and_converted_to_usd() -> None:
    schedule = SwapSchedule(long_pips_per_day=-0.6, short_pips_per_day=0.2)
    common = dict(
        symbol="EURUSD",
        units=100_000,
        price=1.10,
        entry_time=_ny(2024, 1, 10, 10),  # Wednesday
        exit_time=_ny(2024, 1, 11, 10),  # Thursday: triple charge
    )

    long_swap = swap_pnl_usd(schedule, side="LONG", **common)
    short_swap = swap_pnl_usd(schedule, side="SHORT", **common)

    assert long_swap == pytest.approx(3 * -0.6 * 10.0)  # $10 per pip for one EURUSD lot
    assert short_swap == pytest.approx(3 * 0.2 * 10.0)


def test_a_carry_credit_offsets_costs_on_a_high_yielding_side() -> None:
    schedule = SwapSchedule(long_pips_per_day=1.5, short_pips_per_day=-2.0)

    credit = swap_pnl_usd(
        schedule,
        symbol="USDJPY",
        side="LONG",
        units=100_000,
        price=150.0,
        entry_time=_ny(2024, 1, 9, 10),
        exit_time=_ny(2024, 1, 16, 10),
    )

    assert credit == pytest.approx(7 * 1.5 * pip_value_usd("USDJPY", 150.0))
    assert credit > 0
