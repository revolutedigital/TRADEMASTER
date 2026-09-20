"""Tests for the cost layer around the simulator: spread stress, slippage, commission, swap."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.fx import strategy as fx
from app.fx.instruments import ConversionRates, Instrument
from app.fx.sim import core, costs
from app.services.backtest.fx_costs import SwapSchedule, rollover_days_charged, swap_pnl_usd  # noqa: F401

RATES = ConversionRates(
    {"EURUSD": 1.10, "GBPUSD": 1.27, "USDJPY": 150.0, "USDCAD": 1.35, "USDCHF": 0.90}
)
EURUSD = Instrument.from_symbol("EURUSD")
USDJPY = Instrument.from_symbol("USDJPY")
PIP = 0.0001


def bars(n: int = 200, spread_pips: float = 1.0, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0002, n))
    matrix = np.zeros((n, fx.BAR_WIDTH))
    matrix[:, fx.BAR_TIME] = 1_700_000_000 + np.arange(n) * 60.0
    half = spread_pips * PIP / 2
    for offset, sign in ((fx.BID_OPEN, -1), (fx.ASK_OPEN, 1)):
        matrix[:, offset] = np.concatenate(([1.10], close[:-1])) + sign * half
        matrix[:, offset + 1] = close + 0.0003 + sign * half
        matrix[:, offset + 2] = close - 0.0003 + sign * half
        matrix[:, offset + 3] = close + sign * half
    return matrix


def test_widening_scales_the_spread_around_the_mid_and_keeps_the_mid() -> None:
    matrix = bars(spread_pips=1.0)

    wide = costs.widen_spread(matrix, 3.0)

    original_spread = matrix[:, fx.ASK_OPEN] - matrix[:, fx.BID_OPEN]
    new_spread = wide[:, fx.ASK_OPEN] - wide[:, fx.BID_OPEN]
    assert np.allclose(new_spread, 3.0 * original_spread)
    assert np.allclose(
        wide[:, fx.ASK_CLOSE] + wide[:, fx.BID_CLOSE], matrix[:, fx.ASK_CLOSE] + matrix[:, fx.BID_CLOSE]
    )
    assert np.array_equal(wide[:, fx.BAR_TIME], matrix[:, fx.BAR_TIME])
    assert not np.shares_memory(wide, matrix)
    with pytest.raises(ValueError, match="tighter"):
        costs.widen_spread(matrix, 0.5)


def test_fixed_slippage_is_the_same_in_every_bar_and_never_negative() -> None:
    slip = costs.fixed_slippage(10, 0.2, EURUSD)

    assert np.allclose(slip, 0.2 * PIP) and slip.shape == (10,)
    with pytest.raises(ValueError):
        costs.fixed_slippage(10, -0.1, EURUSD)


def test_volatility_slippage_grows_when_recent_bars_were_wide() -> None:
    matrix = bars(400)
    matrix[200:, fx.BID_HIGH] += 0.0010
    matrix[200:, fx.ASK_HIGH] += 0.0010

    slip = costs.volatility_slippage(matrix, EURUSD, base_pips=0.1, range_fraction=0.05)

    assert slip[100] < slip[350]
    assert slip[0] == pytest.approx(0.1 * PIP)  # nothing was known before the first bar


def test_volatility_slippage_never_uses_the_current_or_future_bars() -> None:
    matrix = bars(400)
    baseline = costs.volatility_slippage(matrix, EURUSD, base_pips=0.1, range_fraction=0.05)
    altered = matrix.copy()
    altered[300:, fx.BID_HIGH] += 0.0100
    altered[300:, fx.ASK_HIGH] += 0.0100

    changed = costs.volatility_slippage(altered, EURUSD, base_pips=0.1, range_fraction=0.05)

    assert np.array_equal(baseline[:301], changed[:301])  # bar 300's own range only affects bar 301 on
    assert not np.array_equal(baseline[301:], changed[301:])
    with pytest.raises(ValueError):
        costs.volatility_slippage(matrix, EURUSD, base_pips=-1, range_fraction=0)


def test_commission_per_lot_in_pips_depends_on_the_pair() -> None:
    eurusd = costs.commission_round_trip_pips(
        EURUSD, 1.10, RATES, usd_per_lot_per_side=2.25
    )
    usdjpy = costs.commission_round_trip_pips(
        USDJPY, 150.0, RATES, usd_per_lot_per_side=2.25
    )

    assert eurusd == pytest.approx(0.45)  # $4.50 over $10 a pip
    assert usdjpy == pytest.approx(4.50 / (1000 / 150.0))  # a yen pip is worth less, so it costs more pips


def test_commission_pips_do_not_depend_on_the_size_for_a_per_lot_fee_but_do_with_a_minimum() -> None:
    small = costs.commission_round_trip_pips(EURUSD, 1.10, RATES, usd_per_lot_per_side=2.25, units=1_000)
    large = costs.commission_round_trip_pips(EURUSD, 1.10, RATES, usd_per_lot_per_side=2.25, units=100_000)
    with_minimum = costs.commission_round_trip_pips(
        EURUSD, 1.10, RATES, notional_basis_points=0.2, minimum_usd_per_order=2.0, units=20_000
    )

    assert small == pytest.approx(large)
    assert with_minimum == pytest.approx(2 * 2.0 / (20_000 * PIP))  # the $2 floor: 2.0 pips round trip


def seconds(moment: str) -> float:
    return pd.Timestamp(moment, tz="UTC").timestamp()


CALENDAR = costs.RolloverCalendar(2015, 2032)


def test_the_rollover_calendar_charges_wednesday_triple_and_skips_the_weekend() -> None:
    def days(entry: str, exit_: str) -> int:
        return int(CALENDAR.charged_days(np.array([seconds(entry)]), np.array([seconds(exit_)]))[0])

    assert days("2024-01-10 15:00", "2024-01-11 15:00") == 3   # Wednesday 17:00 NY = 22:00 UTC
    assert days("2024-01-09 15:00", "2024-01-10 15:00") == 1   # Tuesday
    assert days("2024-01-12 15:00", "2024-01-15 15:00") == 1   # Friday rollover; the weekend is free
    assert days("2024-01-09 15:00", "2024-01-16 15:00") == 7
    assert days("2024-01-10 12:00", "2024-01-10 12:30") == 0


def test_the_rollover_follows_new_york_daylight_saving() -> None:
    def days(entry: str, exit_: str) -> int:
        return int(CALENDAR.charged_days(np.array([seconds(entry)]), np.array([seconds(exit_)]))[0])

    assert days("2024-01-09 21:30", "2024-01-09 22:30") == 1  # winter: 17:00 NY is 22:00 UTC
    assert days("2024-07-09 20:30", "2024-07-09 21:30") == 1  # summer: 21:00 UTC
    assert days("2024-07-09 21:30", "2024-07-09 22:30") == 0


def test_calendar_rejects_reversed_times_and_times_outside_its_range() -> None:
    with pytest.raises(ValueError, match="before"):
        CALENDAR.charged_days(np.array([2.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="outside"):
        CALENDAR.charged_days(np.array([seconds("2001-01-01")]), np.array([seconds("2001-01-02")]))
    with pytest.raises(ValueError, match="weekday"):
        costs.RolloverCalendar(triple_weekday=5)


@settings(max_examples=150, deadline=None)
@given(
    start=st.integers(min_value=int(seconds("2020-01-01")), max_value=int(seconds("2025-12-31"))),
    length=st.integers(min_value=0, max_value=14 * 86_400),
)
def test_the_vectorized_calendar_agrees_with_the_reference_implementation(start, length) -> None:
    entry = datetime.fromtimestamp(start, UTC)
    exit_ = datetime.fromtimestamp(start + length, UTC)

    expected = rollover_days_charged(entry, exit_)
    got = CALENDAR.charged_days(np.array([float(start)]), np.array([float(start + length)]))[0]

    assert got == expected


def test_a_scenario_prepares_wider_bars_and_larger_slippage_than_the_base() -> None:
    matrix = bars()

    base_bars, base_slip = costs.prepare_run(matrix, EURUSD, costs.FUSION_ZERO)
    stress_bars, stress_slip = costs.prepare_run(matrix, EURUSD, costs.STRESS)

    assert np.array_equal(base_bars, matrix)
    assert np.all(stress_bars[:, fx.ASK_OPEN] - stress_bars[:, fx.BID_OPEN]
                  >= base_bars[:, fx.ASK_OPEN] - base_bars[:, fx.BID_OPEN])
    assert np.all(stress_slip >= base_slip)
    assert set(costs.SCENARIOS) == {"fusion_zero", "stress", "adverse_swap"}


def run_trades(scenario: costs.CostScenario):
    matrix = bars(3000, spread_pips=0.6, seed=9)
    prepared, slippage = costs.prepare_run(matrix, EURUSD, scenario)
    params = fx.ema_cross_params(fast_span=8, slow_span=21, atr_period=14, stop_atr=1.5, reward_risk=2.0, warmup=60)
    result = core.run_simulation(fx.ema_cross_step, fx.ema_cross_init, params, fx.EMA_CROSS_STATE_SIZE, prepared, slippage)
    commission = costs.commission_round_trip_pips(
        EURUSD, 1.10, RATES, usd_per_lot_per_side=scenario.commission_usd_per_lot_per_side
    )
    return costs.finalize_trades(result, prepared, EURUSD, scenario, commission_pips=commission, calendar=CALENDAR)


def test_finalized_trades_net_out_commission_and_swap_and_express_results_in_r() -> None:
    trades = run_trades(costs.ADVERSE_SWAP)

    assert len(trades) > 10
    assert np.allclose(trades["net_pips"], trades["gross_pips"] - trades["commission_pips"] + trades["swap_pips"])
    assert np.allclose(trades["r_multiple"], trades["net_pips"] / trades["stop_pips"])
    assert (trades["swap_pips"] <= 0).all()
    assert (trades["exit_time"] >= trades["entry_time"]).all()


def test_a_stricter_scenario_never_improves_the_total_result() -> None:
    base = run_trades(costs.FUSION_ZERO)["net_pips"].sum()
    stress = run_trades(costs.STRESS)["net_pips"].sum()

    assert stress <= base
