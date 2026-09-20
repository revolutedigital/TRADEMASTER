"""C1 and C2, the negative controls, checked against independent pandas implementations."""

import numpy as np
import pandas as pd
import pytest

from app.fx import strategy as fx
from app.fx.sim import core
from app.fx.strategies import controls as ct
from tests.unit.fx_strategy_checks import (
    PIP,
    assert_streaming_equals_batch,
    assert_the_future_never_changes_the_past,
    decisions,
    mid_frame,
    simulate,
    synthetic_frame,
)


def blacked_out(index: pd.DatetimeIndex, bar_seconds: int) -> np.ndarray:
    closes = index.tz_convert("America/New_York") + pd.Timedelta(seconds=bar_seconds)
    minutes = closes.hour * 60 + closes.minute
    return np.asarray((minutes >= 16 * 60 + 55) & (minutes < 17 * 60 + 15))


def minute_frame(seed: int) -> pd.DataFrame:
    # Two weeks of one-minute bars that cross the 17:00 New York rollover ten times.
    return synthetic_frame(start="2024-05-13", weeks=2, bar_seconds=60, seed=seed, sigma_pips=0.6)


def five_minute_frame(seed: int) -> pd.DataFrame:
    return synthetic_frame(start="2024-05-13", weeks=6, bar_seconds=300, seed=seed, sigma_pips=1.5)


# --- C1: Bollinger reversion on M1 ---------------------------------------------------------


def expected_bollinger(frame: pd.DataFrame):
    close = mid_frame(frame)["close"]
    mean = close.rolling(20).mean()
    width = 2.0 * close.rolling(20).std(ddof=0)
    short = (close > mean + width) & (width > 0)
    long = (close < mean - width) & (width > 0)
    allowed = ~blacked_out(frame.index, 60)
    fired = ((short | long).to_numpy()) & allowed
    return np.flatnonzero(fired), np.where(short, fx.ENTER_SHORT, fx.ENTER_LONG)[fired]


@pytest.mark.parametrize("seed", [1, 2])
def test_c1_signals_equal_the_independent_pandas_rule(seed: int) -> None:
    frame = minute_frame(seed)
    params = ct.bollinger_reversion_params(PIP)
    matrix = fx.bars_to_matrix(frame)

    intents, stops, targets = decisions(
        ct.bollinger_reversion_step, ct.bollinger_reversion_init, params, ct.BOLLINGER_REVERSION_STATE_SIZE, matrix
    )
    index, sides = expected_bollinger(frame)

    fired = np.flatnonzero(intents != fx.HOLD)
    assert len(index) >= 200
    assert fired.tolist() == index.tolist()
    assert intents[fired].tolist() == sides.tolist()
    assert np.allclose(stops[fired], 3.0 * PIP) and np.allclose(targets[fired], 3.0 * PIP)


def test_c1_never_enters_during_the_rollover_blackout() -> None:
    frame = minute_frame(3)
    intents = decisions(
        ct.bollinger_reversion_step, ct.bollinger_reversion_init, ct.bollinger_reversion_params(PIP),
        ct.BOLLINGER_REVERSION_STATE_SIZE, fx.bars_to_matrix(frame),
    )[0]

    assert not (blacked_out(frame.index, 60) & (intents != fx.HOLD)).any()
    unrestricted = expected_bollinger(frame)[0]
    assert blacked_out(frame.index, 60)[unrestricted].sum() == 0 or len(unrestricted) > 0


def test_c1_closes_a_position_after_thirty_bars_when_nothing_else_happens() -> None:
    frame = minute_frame(4)
    trades = simulate(
        ct.bollinger_reversion_step, ct.bollinger_reversion_init, ct.bollinger_reversion_params(PIP),
        ct.BOLLINGER_REVERSION_STATE_SIZE, fx.bars_to_matrix(frame),
    )

    timed = trades[trades["reason"] == core.EXIT_SIGNAL]
    assert len(trades) > 100 and len(timed) > 0
    assert np.all(timed["exit_index"] - timed["entry_index"] == 30)
    assert np.all(trades["stop_distance"] == 3.0 * PIP)


# --- C2: momentum burst on M5 --------------------------------------------------------------


def expected_burst(frame: pd.DataFrame):
    close = mid_frame(frame)["close"]
    change = close.diff()
    deviation = change.rolling(48).std(ddof=0).shift(1)
    strong = (change.abs() >= 2.5 * deviation) & (deviation > 0)
    fired = strong.to_numpy() & ~blacked_out(frame.index, 300)
    return np.flatnonzero(fired), np.where(change > 0, fx.ENTER_LONG, fx.ENTER_SHORT)[fired]


@pytest.mark.parametrize("seed", [1, 2])
def test_c2_signals_equal_the_independent_pandas_rule(seed: int) -> None:
    frame = five_minute_frame(seed)
    params = ct.momentum_burst_params(PIP)
    matrix = fx.bars_to_matrix(frame)

    intents, stops, targets = decisions(
        ct.momentum_burst_step, ct.momentum_burst_init, params, ct.MOMENTUM_BURST_STATE_SIZE, matrix
    )
    index, sides = expected_burst(frame)

    fired = np.flatnonzero(intents != fx.HOLD)
    assert len(index) >= 100
    assert fired.tolist() == index.tolist()
    assert intents[fired].tolist() == sides.tolist()
    assert np.allclose(stops[fired], 3.0 * PIP) and np.allclose(targets[fired], 3.0 * PIP)


def test_c2_closes_a_position_after_twelve_bars_when_nothing_else_happens() -> None:
    trades = simulate(
        ct.momentum_burst_step, ct.momentum_burst_init, ct.momentum_burst_params(PIP),
        ct.MOMENTUM_BURST_STATE_SIZE, fx.bars_to_matrix(five_minute_frame(3)),
    )

    timed = trades[trades["reason"] == core.EXIT_SIGNAL]
    assert len(trades) > 50 and len(timed) > 0
    assert np.all(timed["exit_index"] - timed["entry_index"] == 12)


def test_the_pip_size_scales_the_stop_and_target_for_a_yen_pair() -> None:
    frame = minute_frame(5)
    yen = ct.bollinger_reversion_params(0.01)
    intents, stops, targets = decisions(
        ct.bollinger_reversion_step, ct.bollinger_reversion_init, yen, ct.BOLLINGER_REVERSION_STATE_SIZE,
        fx.bars_to_matrix(frame),
    )

    fired = intents != fx.HOLD
    assert fired.any()
    assert np.allclose(stops[fired], 0.03) and np.allclose(targets[fired], 0.03)


@pytest.mark.parametrize(
    ("step", "init", "size", "params", "frame_builder"),
    [
        (ct.bollinger_reversion_step, ct.bollinger_reversion_init, ct.BOLLINGER_REVERSION_STATE_SIZE,
         ct.bollinger_reversion_params(PIP), minute_frame),
        (ct.momentum_burst_step, ct.momentum_burst_init, ct.MOMENTUM_BURST_STATE_SIZE,
         ct.momentum_burst_params(PIP), five_minute_frame),
    ],
    ids=["C1", "C2"],
)
def test_streaming_matches_batch_and_the_future_never_changes_the_past(step, init, size, params, frame_builder) -> None:
    matrix = fx.bars_to_matrix(frame_builder(6))

    assert_streaming_equals_batch(step, init, params, size, matrix)
    assert_the_future_never_changes_the_past(step, init, params, size, matrix)


def test_the_params_reject_nonsense() -> None:
    assert ct.bollinger_reversion_params(PIP).shape == (7,)
    with pytest.raises(ValueError):
        ct.bollinger_reversion_params(0.0)
    with pytest.raises(ValueError):
        ct.momentum_burst_params(-1.0)
    with pytest.raises(ValueError):
        ct._control_params(0.0, 3.0, 3.0, PIP, 30, 60)
