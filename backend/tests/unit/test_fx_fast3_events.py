"""Causality, executable fills, barriers, and gaps in the round-3 event factory."""

import numpy as np
import pandas as pd
import pytest

from app.fx import strategy as fx
from app.fx.instruments import Instrument
from app.fx.sim.costs import CostScenario
from scripts.research import fx_fast3_events as events

EURUSD = Instrument.from_symbol("EURUSD")
ZERO_COST = CostScenario("zero", slippage_pips=0.0)


def minute_matrix(minutes: int = 1_200, *, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    close = 1.10 + np.cumsum(rng.normal(0, 0.00004, minutes))
    open_ = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_, close) + 0.00003
    low = np.minimum(open_, close) - 0.00003
    spread = 0.00008 + rng.uniform(0, 0.00003, minutes)
    matrix = np.empty((minutes, fx.BAR_WIDTH))
    timestamps = pd.date_range("2024-01-08", periods=minutes, freq="min", tz="UTC")
    matrix[:, fx.BAR_TIME] = (
        timestamps - pd.Timestamp("1970-01-01", tz="UTC")
    ) / pd.Timedelta(seconds=1)
    for bid_column, ask_column, values in (
        (fx.BID_OPEN, fx.ASK_OPEN, open_),
        (fx.BID_HIGH, fx.ASK_HIGH, high),
        (fx.BID_LOW, fx.ASK_LOW, low),
        (fx.BID_CLOSE, fx.ASK_CLOSE, close),
    ):
        matrix[:, bid_column] = values - spread / 2
        matrix[:, ask_column] = values + spread / 2
    return matrix


def handcrafted_m5(count: int = 8) -> np.ndarray:
    bars = np.empty((count, fx.BAR_WIDTH))
    bars[:, fx.BAR_TIME] = 1_704_067_200 + np.arange(count) * 300
    for index in range(count):
        mid = 1.1000 + index * 0.0004
        bars[index, fx.BID_OPEN] = mid - 0.0001
        bars[index, fx.ASK_OPEN] = mid + 0.0001
        bars[index, fx.BID_CLOSE] = mid + 0.0002 - 0.0001
        bars[index, fx.ASK_CLOSE] = mid + 0.0002 + 0.0001
        bars[index, fx.BID_HIGH] = mid + 0.0003
        bars[index, fx.ASK_HIGH] = mid + 0.0005
        bars[index, fx.BID_LOW] = mid - 0.0003
        bars[index, fx.ASK_LOW] = mid - 0.0001
    return bars


def outcome_features(bars: np.ndarray) -> pd.DataFrame:
    index = pd.to_datetime(bars[:, fx.BAR_TIME] + 300, unit="s", utc=True)
    return pd.DataFrame(
        {"atr_price": np.full(len(bars), 0.001), "history_contiguous": np.ones(len(bars), dtype=bool)},
        index=index,
    )


def test_changing_the_future_cannot_change_a_past_feature_row() -> None:
    original = minute_matrix()
    _, first = events.build_feature_frame(original, EURUSD)
    decision_time = first.index[170]
    changed = original.copy()
    future = changed[:, fx.BAR_TIME] >= decision_time.timestamp()
    changed[future, 1:] += np.linspace(0.0, 0.25, future.sum())[:, None]

    _, second = events.build_feature_frame(changed, EURUSD)

    assert np.isfinite(first.loc[decision_time, "return_72"])
    assert np.isfinite(first.loc[decision_time, "h1_trend_atr"])
    pd.testing.assert_series_equal(first.loc[decision_time], second.loc[decision_time])


def test_coarse_context_uses_only_bars_that_have_fully_closed() -> None:
    original = minute_matrix()
    _, first = events.build_feature_frame(original, EURUSD)
    decision_time = pd.Timestamp("2024-01-08 15:35", tz="UTC")
    changed = original.copy()
    minute_times = pd.to_datetime(changed[:, fx.BAR_TIME], unit="s", utc=True)
    still_open_h1 = (minute_times >= pd.Timestamp("2024-01-08 15:35", tz="UTC")) & (
        minute_times < pd.Timestamp("2024-01-08 16:00", tz="UTC")
    )
    changed[still_open_h1, 1:] += 0.1

    _, second = events.build_feature_frame(changed, EURUSD)

    assert first.loc[decision_time, "h1_trend_atr"] == second.loc[decision_time, "h1_trend_atr"]


def test_next_open_bid_ask_fill_and_terminal_exit_are_used() -> None:
    bars = handcrafted_m5()
    result = events.build_outcome_frame(
        bars,
        outcome_features(bars),
        EURUSD,
        events.EventCosts(0.0, 0.0, ZERO_COST, ZERO_COST),
        horizons_minutes=(15,),
    )
    long = result[(result["decision_index"] == 0) & (result["side"] == fx.LONG)].iloc[0]

    expected_entry = bars[1, fx.ASK_OPEN]
    expected_exit = bars[3, fx.BID_CLOSE]
    assert long["entry_index"] == 1
    assert long["exit_index"] == 3
    assert long["terminal_r_base"] == pytest.approx((expected_exit - expected_entry) / 0.001)


def test_stop_wins_when_stop_and_target_are_touched_inside_the_same_bar() -> None:
    bars = handcrafted_m5()
    entry = bars[1, fx.ASK_OPEN]
    bars[2, fx.BID_LOW] = entry - 0.0011
    bars[2, fx.BID_HIGH] = entry + 0.0021
    bars[2, fx.ASK_HIGH] = bars[2, fx.BID_HIGH] + 0.0002

    result = events.build_outcome_frame(
        bars,
        outcome_features(bars),
        EURUSD,
        events.EventCosts(0.0, 0.0, ZERO_COST, ZERO_COST),
        horizons_minutes=(15,),
    )
    long = result[(result["decision_index"] == 0) & (result["side"] == fx.LONG)].iloc[0]

    assert long["target_0_5r_before_stop"] == 0
    assert long["target_1_0r_before_stop"] == 0
    assert long["target_2_0r_before_stop"] == 0


def test_an_event_whose_horizon_crosses_a_gap_is_removed() -> None:
    bars = np.delete(handcrafted_m5(), 2, axis=0)
    result = events.build_outcome_frame(
        bars,
        outcome_features(bars),
        EURUSD,
        events.EventCosts(0.0, 0.0, ZERO_COST, ZERO_COST),
        horizons_minutes=(15,),
    )

    assert not ((result["decision_index"] == 0) & (result["side"] == fx.LONG)).any()


def test_directional_features_flip_signed_context_but_not_costs() -> None:
    frame = pd.DataFrame({"return_3": [0.2], "spread_pips": [0.8]})

    short = events.directional_features(frame, fx.SHORT)

    assert short.loc[0, "return_3"] == -0.2
    assert short.loc[0, "spread_pips"] == 0.8
    assert short.loc[0, "side"] == fx.SHORT


def test_compiled_wide_outcomes_match_the_auditable_long_form() -> None:
    bars = handcrafted_m5()
    features = outcome_features(bars)
    costs = events.EventCosts(0.0, 0.0, ZERO_COST, ZERO_COST)
    long_form = events.build_outcome_frame(
        bars, features, EURUSD, costs, horizons_minutes=(15,)
    )
    wide = events.build_outcome_wide(bars, features, EURUSD, costs, horizons_minutes=(15,))

    for side, side_name in ((fx.LONG, "long"), (fx.SHORT, "short")):
        expected = long_form[
            (long_form["decision_index"] == 0) & (long_form["side"] == side)
        ].iloc[0]
        for field in (
            "risk_price",
            "terminal_r_base",
            "terminal_r_stress",
            "mfe_r_base",
            "mae_r_base",
            "target_0_5r_before_stop",
            "target_1_0r_before_stop",
            "target_2_0r_before_stop",
        ):
            assert wide.iloc[0][f"h15_{side_name}_{field}"] == pytest.approx(expected[field])


def test_barrier_return_charges_commission_and_stop_slippage() -> None:
    bars = handcrafted_m5()
    entry = bars[1, fx.ASK_OPEN]
    bars[2, fx.BID_LOW] = entry - 0.003
    features = outcome_features(bars)
    slippage = CostScenario("slip", slippage_pips=0.2)
    costs = events.EventCosts(0.5, 0.5, slippage, slippage)

    wide = events.build_outcome_wide(
        bars, features, EURUSD, costs, horizons_minutes=(15,)
    )

    risk = wide.iloc[0]["h15_long_risk_price"]
    expected = -1.0 - (0.5 * EURUSD.pip_size) / risk - (0.2 * EURUSD.pip_size) / risk
    assert wide.iloc[0]["h15_long_barrier_1_0r_base"] == pytest.approx(expected)
