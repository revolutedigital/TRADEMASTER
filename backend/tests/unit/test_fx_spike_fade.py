"""F3: the spike fade, from hand-built spikes and from an independent pandas implementation."""

import numpy as np
import pandas as pd
import pytest

from app.fx import strategy as fx
from app.fx.sim import core
from app.fx.strategies import spike_fade as sf
from tests.unit.fx_strategy_checks import (
    PIP,
    assert_streaming_equals_batch,
    assert_the_future_never_changes_the_past,
    decisions,
    mid_frame,
    simulate,
    synthetic_frame,
)

STATE = sf.SPIKE_FADE_STATE_SIZE
QUIET_RANGE_PIPS = 2.0  # a 1 pip step plus a 0.5 pip wick on each side


def frame_from_steps(
    index: pd.DatetimeIndex,
    steps_pips: np.ndarray,
    wick_pips: np.ndarray | float = 0.5,
    wide_spread_at: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Bars whose mid moves by `steps_pips` from one bar to the next, with wicks and a 0.4 pip spread."""
    count = len(index)
    wick = np.broadcast_to(np.asarray(wick_pips, dtype=float), (count,))
    close = 1.10 + np.cumsum(steps_pips) * PIP
    open_ = np.concatenate(([1.10], close[:-1]))
    high = np.maximum(open_, close) + wick * PIP
    low = np.minimum(open_, close) - wick * PIP
    half = np.full(count, 0.2 * PIP)
    for position, spread_pips in (wide_spread_at or {}).items():
        half[position] = 0.5 * spread_pips * PIP
    frame = pd.DataFrame(index=index)
    for side, sign in (("bid", -1), ("ask", 1)):
        frame[f"{side}_open"] = open_ + sign * half
        frame[f"{side}_high"] = high + sign * half
        frame[f"{side}_low"] = low + sign * half
        frame[f"{side}_close"] = close + sign * half
    return frame


def quiet_steps(count: int) -> np.ndarray:
    return np.where(np.arange(count) % 2 == 0, 1.0, -1.0)


def hand_built(spike_step: float, *, spike_at: int = 100, start: str = "2024-05-13 06:00", **kwargs):
    index = pd.date_range(start, periods=160, freq="5min", tz="UTC")
    steps = quiet_steps(160)
    steps[spike_at] = spike_step
    wick = np.full(160, 0.5)
    wick[spike_at] = 1.0
    return frame_from_steps(index, steps, wick, **kwargs), spike_at


def run(frame: pd.DataFrame, params: np.ndarray, position: int = fx.FLAT):
    matrix = fx.bars_to_matrix(frame)
    return matrix, decisions(sf.spike_fade_step, sf.spike_fade_init, params, STATE, matrix, position)


def test_a_spike_up_is_faded_with_the_pre_registered_stop_and_target() -> None:
    frame, at = hand_built(+10.0)
    matrix, (intents, stops, targets) = run(frame, sf.spike_fade_4_params())

    mid = mid_frame(frame).iloc[at]
    spike_range = mid["high"] - mid["low"]
    assert np.flatnonzero(intents != fx.HOLD).tolist() == [at]
    assert intents[at] == fx.ENTER_SHORT
    assert stops[at] == pytest.approx(mid["high"] + 0.5 * spike_range - mid["close"], abs=1e-12)
    assert targets[at] == pytest.approx(mid["close"] - (mid["high"] - 0.5 * spike_range), abs=1e-12)
    assert stops[at] > 0 and targets[at] > 0


def test_a_spike_down_is_bought() -> None:
    frame, at = hand_built(-10.0)
    _, (intents, stops, targets) = run(frame, sf.spike_fade_4_params())

    mid = mid_frame(frame).iloc[at]
    spike_range = mid["high"] - mid["low"]
    assert np.flatnonzero(intents != fx.HOLD).tolist() == [at]
    assert intents[at] == fx.ENTER_LONG
    assert stops[at] == pytest.approx(mid["close"] - (mid["low"] - 0.5 * spike_range), abs=1e-12)
    assert targets[at] == pytest.approx((mid["low"] + 0.5 * spike_range) - mid["close"], abs=1e-12)


def test_the_threshold_separates_the_two_variants() -> None:
    # A range of 10 pips is 5 times the quiet 2 pips: enough for threshold 4, not for 6.
    frame, at = hand_built(+8.0)
    assert QUIET_RANGE_PIPS * 5 == pytest.approx(10.0)

    fired_4 = run(frame, sf.spike_fade_4_params())[1][0]
    fired_6 = run(frame, sf.spike_fade_6_params())[1][0]

    assert fired_4[at] == fx.ENTER_SHORT
    assert np.all(fired_6 == fx.HOLD)


def test_a_bar_with_a_long_wick_and_a_small_body_is_not_a_spike() -> None:
    index = pd.date_range("2024-05-13 06:00", periods=160, freq="5min", tz="UTC")
    steps = quiet_steps(160)
    steps[100] = 2.0
    wick = np.full(160, 0.5)
    wick[100] = 6.0  # range 14 pips, body 2 pips: well under 60% of the range
    frame = frame_from_steps(index, steps, wick)

    intents = run(frame, sf.spike_fade_4_params())[1][0]

    assert np.all(intents == fx.HOLD)


def test_a_blown_out_spread_blocks_the_entry() -> None:
    frame, at = hand_built(+10.0, wide_spread_at={100: 5.0})

    intents = run(frame, sf.spike_fade_4_params())[1][0]

    assert np.all(intents == fx.HOLD)


@pytest.mark.parametrize(
    ("close_clock", "allowed"),
    [("16:50", True), ("16:55", False), ("17:00", False), ("17:10", False), ("17:15", True)],
)
def test_no_entry_while_the_new_york_rollover_is_blacked_out(close_clock: str, allowed: bool) -> None:
    # Bar 100 opens 5 minutes before its close, which is the New York time in the parametrize.
    close_ny = pd.Timestamp(f"2024-05-15 {close_clock}", tz="America/New_York")
    start = close_ny.tz_convert("UTC") - pd.Timedelta(minutes=5 * 101)
    frame, at = hand_built(+10.0, start=str(start.tz_localize(None)))

    intents = run(frame, sf.spike_fade_4_params())[1][0]

    assert bool(intents[at] != fx.HOLD) == allowed


def test_a_position_is_closed_after_exactly_twelve_bars_when_nothing_else_happens() -> None:
    frame, at = hand_built(+10.0)
    matrix = fx.bars_to_matrix(frame)

    trades = simulate(sf.spike_fade_step, sf.spike_fade_init, sf.spike_fade_4_params(), STATE, matrix)

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["side"] == fx.SHORT
    assert trade["entry_index"] == at + 1
    assert trade["exit_index"] == trade["entry_index"] + 12
    assert trade["reason"] == core.EXIT_SIGNAL


def test_the_target_is_hit_when_the_spike_retraces() -> None:
    frame, at = hand_built(+10.0)
    steps = quiet_steps(160)
    steps[at] = 10.0
    steps[at + 2] = -8.0  # a retracement well beyond half of the spike range
    wick = np.full(160, 0.5)
    wick[at] = 1.0
    frame = frame_from_steps(frame.index, steps, wick)

    trades = simulate(sf.spike_fade_step, sf.spike_fade_init, sf.spike_fade_4_params(), STATE, fx.bars_to_matrix(frame))

    first = trades.iloc[0]  # the retracement bar is itself a spike, so a second fade follows
    assert first["entry_index"] == at + 1 and first["side"] == fx.SHORT
    assert first["reason"] == core.EXIT_TARGET


def test_the_stop_is_hit_when_the_spike_keeps_going() -> None:
    frame, at = hand_built(+10.0)
    steps = quiet_steps(160)
    steps[at] = 10.0
    steps[at + 2] = 9.0  # the move continues past the stop half a range beyond the extreme
    wick = np.full(160, 0.5)
    wick[at] = 1.0
    frame = frame_from_steps(frame.index, steps, wick)

    trades = simulate(sf.spike_fade_step, sf.spike_fade_init, sf.spike_fade_4_params(), STATE, fx.bars_to_matrix(frame))

    first = trades.iloc[0]
    assert first["entry_index"] == at + 1 and first["side"] == fx.SHORT
    assert first["reason"] in (core.EXIT_STOP, core.EXIT_STOP_GAP)
    assert first["stop_distance"] > 0


def random_spiky_frame(seed: int) -> pd.DataFrame:
    base = synthetic_frame(start="2024-03-04", weeks=8, bar_seconds=300, seed=seed)
    rng = np.random.default_rng(seed)
    count = len(base)
    steps = rng.normal(0.0, 1.2, count) * np.where(rng.random(count) < 0.02, 9.0, 1.0)
    return frame_from_steps(base.index, steps, np.abs(rng.normal(0.0, 0.6, count)))


def expected_spikes(frame: pd.DataFrame, threshold: float):
    mid = mid_frame(frame)
    bar_range = mid["high"] - mid["low"]
    body = mid["close"] - mid["open"]
    spread = frame["ask_close"] - frame["bid_close"]
    typical_range = bar_range.rolling(48).median().shift(1)
    typical_spread = spread.rolling(48).median().shift(1)
    closes = mid.index.tz_convert("America/New_York") + pd.Timedelta(minutes=5)
    minutes = closes.hour * 60 + closes.minute
    blacked_out = (minutes >= 16 * 60 + 55) & (minutes < 17 * 60 + 15)
    spike = (
        typical_range.notna()
        & (bar_range > 0)
        & (bar_range >= threshold * typical_range)
        & (body.abs() >= 0.6 * bar_range)
        & (spread <= 2.0 * typical_spread)
        & ~blacked_out
    )
    up = body > 0
    stop = np.where(up, mid["high"] + 0.5 * bar_range - mid["close"], mid["close"] - (mid["low"] - 0.5 * bar_range))
    target = np.where(up, mid["close"] - (mid["high"] - 0.5 * bar_range), (mid["low"] + 0.5 * bar_range) - mid["close"])
    valid = spike.to_numpy() & (stop > 0) & (target > 0)
    return np.flatnonzero(valid), np.where(up, fx.ENTER_SHORT, fx.ENTER_LONG)[valid], stop[valid], target[valid]


@pytest.mark.parametrize(("params", "threshold"), [(sf.spike_fade_4_params(), 4.0), (sf.spike_fade_6_params(), 6.0)])
@pytest.mark.parametrize("seed", [1, 2])
def test_the_signals_equal_the_independent_pandas_rule(params, threshold, seed) -> None:
    frame = random_spiky_frame(seed)
    matrix, (intents, stops, targets) = run(frame, params)

    index, sides, stop, target = expected_spikes(frame, threshold)

    fired = np.flatnonzero(intents != fx.HOLD)
    assert len(index) >= 8
    assert fired.tolist() == index.tolist()
    assert intents[fired].tolist() == sides.tolist()
    assert np.allclose(stops[fired], stop, rtol=0, atol=1e-12)
    assert np.allclose(targets[fired], target, rtol=0, atol=1e-12)


def test_the_higher_threshold_fires_on_a_subset_of_the_lower_one() -> None:
    frame = random_spiky_frame(3)

    low = set(np.flatnonzero(run(frame, sf.spike_fade_4_params())[1][0] != fx.HOLD))
    high = set(np.flatnonzero(run(frame, sf.spike_fade_6_params())[1][0] != fx.HOLD))

    assert high < low


@pytest.mark.parametrize("params", [sf.spike_fade_4_params(), sf.spike_fade_6_params()], ids=["F3a", "F3b"])
def test_streaming_matches_batch_and_the_future_never_changes_the_past(params) -> None:
    matrix = fx.bars_to_matrix(random_spiky_frame(4))

    assert_streaming_equals_batch(sf.spike_fade_step, sf.spike_fade_init, params, STATE, matrix)
    assert_the_future_never_changes_the_past(sf.spike_fade_step, sf.spike_fade_init, params, STATE, matrix)


def test_it_never_enters_while_a_position_is_open() -> None:
    frame = random_spiky_frame(5)

    intents = run(frame, sf.spike_fade_4_params(), position=fx.SHORT)[1][0]

    assert fx.ENTER_LONG not in intents and fx.ENTER_SHORT not in intents


def test_the_params_reject_nonsense() -> None:
    assert sf.spike_fade_params(threshold=4.0).shape == (9,)
    for override in (
        dict(threshold=1.0),
        dict(body_fraction=0.0),
        dict(retrace=1.0),
        dict(stop_range=0.0),
        dict(max_bars=0),
        dict(bar_seconds=30),
        dict(spread_multiple=0.0),
    ):
        with pytest.raises(ValueError):
            sf.spike_fade_params(**{"threshold": 4.0, **override})


def test_the_stop_and_target_levels_are_those_of_the_spike_when_anchored_to_the_signal() -> None:
    frame, at = hand_built(+10.0)
    steps = quiet_steps(160)
    steps[at] = 10.0
    steps[at + 2] = -8.0
    wick = np.full(160, 0.5)
    wick[at] = 1.0
    frame = frame_from_steps(frame.index, steps, wick)
    mid = mid_frame(frame).iloc[at]
    spike_range = mid["high"] - mid["low"]

    trades = simulate(
        sf.spike_fade_step, sf.spike_fade_init, sf.spike_fade_4_params(), STATE, fx.bars_to_matrix(frame),
        anchor_signal=True, max_entry_gap=300.0,
    )

    first = trades.iloc[0]
    assert first["reason"] == core.EXIT_TARGET
    assert first["exit_price"] == pytest.approx(mid["high"] - 0.5 * spike_range, abs=1e-12)
    assert first["entry_price"] + first["stop_distance"] == pytest.approx(mid["high"] + 0.5 * spike_range, abs=1e-12)


def widen_at(frame: pd.DataFrame, at: int, *, close_pips: float | None = None, open_pips: float | None = None) -> pd.DataFrame:
    """Set the bid/ask spread of one bar at its close and/or its open, leaving the mid alone."""
    frame = frame.copy()
    for pips, column in ((close_pips, "close"), (open_pips, "open")):
        if pips is not None:
            mid = 0.5 * (frame[f"bid_{column}"].iloc[at] + frame[f"ask_{column}"].iloc[at])
            frame.iloc[at, frame.columns.get_loc(f"bid_{column}")] = mid - 0.5 * pips * PIP
            frame.iloc[at, frame.columns.get_loc(f"ask_{column}")] = mid + 0.5 * pips * PIP
    return frame


@pytest.mark.parametrize(("closing_spread", "fires"), [(0.78, True), (0.82, False)])
def test_the_spread_limit_is_twice_the_median_closing_spread(closing_spread: float, fires: bool) -> None:
    frame, at = hand_built(+10.0)  # the quiet spread is 0.4 pip, so the limit is 0.8 pip

    intents = run(widen_at(frame, at, close_pips=closing_spread), sf.spike_fade_4_params())[1][0]

    assert bool(intents[at] != fx.HOLD) is fires


def test_only_the_closing_spread_counts_not_the_opening_one() -> None:
    frame, at = hand_built(+10.0)

    intents = run(widen_at(frame, at, open_pips=6.0), sf.spike_fade_4_params())[1][0]

    assert intents[at] == fx.ENTER_SHORT


@pytest.mark.parametrize(("body_share", "fires"), [(0.59, False), (0.61, True)])
def test_the_body_must_be_at_least_sixty_percent_of_the_range(body_share: float, fires: bool) -> None:
    index = pd.date_range("2024-05-13 06:00", periods=160, freq="5min", tz="UTC")
    steps = quiet_steps(160)
    body = 6.0
    steps[100] = body
    wick = np.full(160, 0.5)
    wick[100] = (body / body_share - body) / 2  # the range is body / share, the wicks share the rest
    frame = frame_from_steps(index, steps, wick)

    intents = run(frame, sf.spike_fade_4_params())[1][0]

    assert bool(intents[100] != fx.HOLD) is fires


def test_the_signals_equal_the_pandas_rule_with_gaps_and_variable_spreads() -> None:
    base = synthetic_frame(start="2024-03-04", weeks=8, bar_seconds=300, seed=41, spread_jitter=True)
    rng = np.random.default_rng(41)
    steps = rng.normal(0.0, 1.2, len(base)) * np.where(rng.random(len(base)) < 0.02, 9.0, 1.0)
    frame = frame_from_steps(base.index, steps, np.abs(rng.normal(0.0, 0.6, len(base))))
    jitter = 0.5 + 2.5 * rng.random(len(base))
    for column in ("open", "high", "low", "close"):
        mid = 0.5 * (frame[f"bid_{column}"] + frame[f"ask_{column}"])
        frame[f"bid_{column}"] = mid - 0.2 * PIP * jitter
        frame[f"ask_{column}"] = mid + 0.2 * PIP * jitter
    matrix, (intents, stops, targets) = run(frame, sf.spike_fade_4_params())

    index, sides, stop, target = expected_spikes(frame, 4.0)

    fired = np.flatnonzero(intents != fx.HOLD)
    assert len(index) >= 8 and fired.tolist() == index.tolist()
    assert np.allclose(stops[fired], stop, rtol=0, atol=1e-12) and np.allclose(targets[fired], target, rtol=0, atol=1e-12)
