"""F5: mean reversion of a cross, checked against pandas, plus the synthetic cross construction."""

import numpy as np
import pandas as pd
import pytest

from app.fx import strategy as fx
from app.fx.sim import core
from app.fx.strategies import pairs_spread as ps
from tests.unit.fx_strategy_checks import (
    PIP,
    assert_streaming_equals_batch,
    assert_the_future_never_changes_the_past,
    decisions,
    mid_frame,
    simulate,
    synthetic_frame,
)

STATE = ps.PAIRS_SPREAD_STATE_SIZE
PARAMS = ps.pairs_spread_params()


def oscillating_frame(seed: int) -> pd.DataFrame:
    return synthetic_frame(
        start="2024-01-01", weeks=26, bar_seconds=3600, seed=seed, sigma_pips=8.0, mean_reversion=0.05
    )


def flat_frame(jump_pips: float, *, bars: int = 700, at: int = 500) -> pd.DataFrame:
    """A market oscillating by 0.5 pip around 1.10 + 0.5 pip (z-score unit: 0.5 pip), with one jump.

    After bar `at` the price stays put `jump_pips` above the old mean, so its z-score is jump / 0.5.
    """
    index = pd.date_range("2024-01-01", periods=bars, freq="1h", tz="UTC")
    mid = np.where(np.arange(bars) % 2 == 0, 1.10, 1.10 + 1.0 * PIP)
    mid[at:] = 1.10 + 0.5 * PIP + jump_pips * PIP
    frame = pd.DataFrame(index=index)
    for side, sign in (("bid", -1), ("ask", 1)):
        for column in ("open", "high", "low", "close"):
            frame[f"{side}_{column}"] = mid + sign * 0.2 * PIP
    return frame


def expected_signals(frame: pd.DataFrame, entry_z=2.0, stop_z=3.5):
    close = mid_frame(frame)["close"]
    mean = close.rolling(480).mean().shift(1)
    deviation = close.rolling(480).std(ddof=0).shift(1)
    z = (close - mean) / deviation
    short = (z >= entry_z) & (z < stop_z)
    long = (z <= -entry_z) & (z > -stop_z)
    stop = np.where(short, mean + stop_z * deviation - close, close - (mean - stop_z * deviation))
    target = np.where(short, close - mean, mean - close)
    fired = (short | long).to_numpy()
    return np.flatnonzero(fired), np.where(short, fx.ENTER_SHORT, fx.ENTER_LONG)[fired], stop[fired], target[fired]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_signals_equal_the_independent_pandas_rule(seed: int) -> None:
    frame = oscillating_frame(seed)
    matrix = fx.bars_to_matrix(frame)

    intents, stops, targets = decisions(ps.pairs_spread_step, ps.pairs_spread_init, PARAMS, STATE, matrix)
    index, sides, stop, target = expected_signals(frame)

    fired = np.flatnonzero(intents != fx.HOLD)
    assert len(index) >= 20
    assert fired.tolist() == index.tolist()
    assert intents[fired].tolist() == sides.tolist()
    assert np.allclose(stops[fired], stop, rtol=0, atol=1e-12)
    assert np.allclose(targets[fired], target, rtol=0, atol=1e-12)
    assert np.all(stops[fired] > 0) and np.all(targets[fired] > 0)


def test_an_overextended_price_is_faded_and_a_price_already_past_the_stop_is_not() -> None:
    inside = decisions(
        ps.pairs_spread_step, ps.pairs_spread_init, PARAMS, STATE, fx.bars_to_matrix(flat_frame(1.2))
    )[0]
    beyond = decisions(
        ps.pairs_spread_step, ps.pairs_spread_init, PARAMS, STATE, fx.bars_to_matrix(flat_frame(3.0))
    )[0]
    down = decisions(
        ps.pairs_spread_step, ps.pairs_spread_init, PARAMS, STATE, fx.bars_to_matrix(flat_frame(-1.2))
    )[0]

    assert inside[500] == fx.ENTER_SHORT
    assert down[500] == fx.ENTER_LONG
    assert beyond[500] == fx.HOLD  # |z| is already past the 3.5 stop, so the stop would be behind us


def test_nothing_is_decided_until_the_window_is_full() -> None:
    intents = decisions(
        ps.pairs_spread_step, ps.pairs_spread_init, PARAMS, STATE, fx.bars_to_matrix(oscillating_frame(4))
    )[0]

    assert np.all(intents[:480] == fx.HOLD)
    assert (intents[480:] != fx.HOLD).any()


def test_the_position_is_closed_after_exactly_120_bars_when_nothing_else_happens() -> None:
    # The price jumps to z = 2.4 and then stays put: neither the target (the old mean) nor the
    # stop is touched, so only the time limit ends the trade.
    frame = flat_frame(1.2)
    matrix = fx.bars_to_matrix(frame)

    trades = simulate(ps.pairs_spread_step, ps.pairs_spread_init, PARAMS, STATE, matrix)

    first = trades.iloc[0]
    assert first["side"] == fx.SHORT and first["entry_index"] == 501
    assert first["exit_index"] == first["entry_index"] + 120
    assert first["reason"] == core.EXIT_SIGNAL


def test_a_mean_reverting_market_mostly_ends_at_the_target_and_stops_lose() -> None:
    frame = oscillating_frame(5)
    matrix = fx.bars_to_matrix(frame)

    trades = simulate(ps.pairs_spread_step, ps.pairs_spread_init, PARAMS, STATE, matrix)

    pnl = (trades["exit_price"] - trades["entry_price"]) * trades["side"]
    targets = trades["reason"] == core.EXIT_TARGET
    stops = trades["reason"].isin([core.EXIT_STOP, core.EXIT_STOP_GAP])
    assert len(trades) >= 20
    assert targets.sum() > stops.sum() > 0
    assert np.all(pnl[targets] > 0) and np.all(pnl[stops] < 0)
    assert np.all(trades["stop_distance"] > 0)


@pytest.mark.parametrize("seed", [6, 7])
def test_streaming_matches_batch_and_the_future_never_changes_the_past(seed: int) -> None:
    matrix = fx.bars_to_matrix(oscillating_frame(seed))

    assert_streaming_equals_batch(ps.pairs_spread_step, ps.pairs_spread_init, PARAMS, STATE, matrix)
    assert_the_future_never_changes_the_past(ps.pairs_spread_step, ps.pairs_spread_init, PARAMS, STATE, matrix)


def test_the_params_reject_nonsense() -> None:
    assert ps.pairs_spread_params().shape == (3,)
    for bad in (dict(entry_z=0.0), dict(entry_z=4.0), dict(max_bars=0), dict(stop_z=1.0)):
        with pytest.raises(ValueError):
            ps.pairs_spread_params(**bad)


def leg(seed: int, minutes: int = 600) -> np.ndarray:
    frame = synthetic_frame(start="2024-05-13", weeks=1, bar_seconds=60, seed=seed)
    return fx.bars_to_matrix(frame.iloc[:minutes])


def test_the_synthetic_cross_pays_the_spread_of_both_legs() -> None:
    numerator, denominator = leg(1), leg(2)

    cross = ps.synthetic_cross(numerator, denominator)

    assert cross.shape == numerator.shape
    assert np.allclose(cross[:, fx.ASK_OPEN], numerator[:, fx.ASK_OPEN] / denominator[:, fx.BID_OPEN])
    assert np.allclose(cross[:, fx.BID_CLOSE], numerator[:, fx.BID_CLOSE] / denominator[:, fx.ASK_CLOSE])
    assert np.all(cross[:, fx.ASK_CLOSE] > cross[:, fx.BID_CLOSE])
    leg_spread = (numerator[:, fx.ASK_CLOSE] - numerator[:, fx.BID_CLOSE]) / numerator[:, fx.BID_CLOSE]
    cross_spread = (cross[:, fx.ASK_CLOSE] - cross[:, fx.BID_CLOSE]) / cross[:, fx.BID_CLOSE]
    assert np.all(cross_spread > leg_spread)  # crossing two spreads costs more than crossing one


def test_the_synthetic_cross_keeps_only_the_minutes_both_legs_quoted() -> None:
    numerator, denominator = leg(3), leg(4)
    numerator = np.delete(numerator, [10, 11, 200], axis=0)
    denominator = np.delete(denominator, [50, 200, 201, 202], axis=0)

    cross = ps.synthetic_cross(numerator, denominator)

    assert cross.shape[0] == 600 - 6
    assert np.all(np.diff(cross[:, fx.BAR_TIME]) >= 60)
    assert np.all(cross[:, fx.BID_HIGH] >= cross[:, fx.BID_LOW])
    assert np.all(cross[:, fx.ASK_HIGH] >= cross[:, fx.ASK_LOW])
    with pytest.raises(ValueError, match="9"):
        ps.synthetic_cross(numerator[:, :5], denominator)
