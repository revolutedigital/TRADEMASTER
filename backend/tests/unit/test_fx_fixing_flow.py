"""F2: the fixing flow is a clock, checked against an independent pandas implementation."""

import numpy as np
import pandas as pd
import pytest

from app.fx import sessions
from app.fx import strategy as fx
from app.fx.sim import core
from app.fx.strategies import fixing_flow as ff
from app.fx.strategies.common import FAR_TARGET
from tests.unit.fx_strategy_checks import (
    assert_streaming_equals_batch,
    assert_the_future_never_changes_the_past,
    decisions,
    mid_frame,
    simulate,
    synthetic_frame,
)

STATE = ff.FIXING_FLOW_STATE_SIZE


def frame_for_tests(seed: int = 1) -> pd.DataFrame:
    # 2024-03-04 to 2024-04-29 crosses the US switch (Mar 10) and the European one (Mar 31).
    return synthetic_frame(start="2024-03-04", weeks=8, bar_seconds=300, seed=seed)


def expected_entries(frame: pd.DataFrame, entry_clock: str) -> tuple[np.ndarray, np.ndarray]:
    """Bars whose close is the entry time, and 3 x the mean true range of the last 14 bars."""
    mid = mid_frame(frame)
    previous_close = mid["close"].shift(1)
    true_range = pd.concat(
        [mid["high"] - mid["low"], (mid["high"] - previous_close).abs(), (mid["low"] - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(14).mean()
    local = mid.index.tz_convert("Europe/London")
    closes_at = local.hour * 3600 + local.minute * 60 + 300
    signal = (closes_at == sessions.seconds_of_day(entry_clock)) & atr.notna().to_numpy()
    return np.flatnonzero(signal), 3.0 * atr.to_numpy()[signal]


@pytest.mark.parametrize(
    ("pair", "builder", "entry_clock", "side"),
    [
        ("USDJPY", ff.pre_fixing_params, "15:00", fx.ENTER_LONG),
        ("EURUSD", ff.pre_fixing_params, "15:00", fx.ENTER_SHORT),
        ("USDCHF", ff.post_fixing_params, "16:05", fx.ENTER_SHORT),
        ("AUDUSD", ff.post_fixing_params, "16:05", fx.ENTER_LONG),
    ],
)
def test_the_signals_equal_the_independent_pandas_rule(pair, builder, entry_clock, side) -> None:
    frame = frame_for_tests(1)
    matrix = fx.bars_to_matrix(frame)
    params = builder(pair)

    intents, stops, targets = decisions(ff.fixing_flow_step, ff.fixing_flow_init, params, STATE, matrix)
    expected_index, expected_stop = expected_entries(frame, entry_clock)

    fired = np.flatnonzero(intents != fx.HOLD)
    assert len(expected_index) >= 30
    assert fired.tolist() == expected_index.tolist()
    assert np.all(intents[fired] == side)
    assert np.allclose(stops[fired], expected_stop, rtol=0, atol=1e-12)
    assert np.all(targets[fired] == FAR_TARGET)


def test_the_dollar_direction_follows_which_side_of_the_pair_the_dollar_is_on() -> None:
    assert ff.pair_side_for_dollar("USDJPY", buy_dollar=True) == fx.LONG
    assert ff.pair_side_for_dollar("USDJPY", buy_dollar=False) == fx.SHORT
    assert ff.pair_side_for_dollar("EURUSD", buy_dollar=True) == fx.SHORT
    assert ff.pair_side_for_dollar("EURUSD", buy_dollar=False) == fx.LONG
    with pytest.raises(ValueError, match="dollar"):
        ff.pair_side_for_dollar("EURGBP", buy_dollar=True)
    sides = {pair: ff.pair_side_for_dollar(pair, buy_dollar=True) for pair in ff.USD_BASE | ff.USD_QUOTE}
    assert len(sides) == 7


def test_the_local_hour_follows_daylight_saving_in_london() -> None:
    frame = frame_for_tests(2)
    matrix = fx.bars_to_matrix(frame)
    intents = decisions(ff.fixing_flow_step, ff.fixing_flow_init, ff.pre_fixing_params("USDJPY"), STATE, matrix)[0]

    fired = frame.index[np.flatnonzero(intents != fx.HOLD)]
    opens = fired + pd.Timedelta(minutes=5)  # the signal is the bar before the 15:00 open
    london = opens.tz_convert("Europe/London")

    assert np.all(london.hour * 60 + london.minute == 15 * 60)
    assert len(set(opens.hour)) == 2  # 14:00 UTC in winter, 13:00 UTC in summer: the clock moved


def test_trades_enter_at_fifteen_and_leave_at_fifty_five_past_with_a_wide_stop() -> None:
    frame = frame_for_tests(3)
    matrix = fx.bars_to_matrix(frame)
    params = ff.pre_fixing_params("USDJPY")

    trades = simulate(ff.fixing_flow_step, ff.fixing_flow_init, params, STATE, matrix)

    opened = frame.index[trades["entry_index"]].tz_convert("Europe/London")
    closed = frame.index[trades["exit_index"]].tz_convert("Europe/London")
    assert len(trades) >= 30
    assert np.all(opened.hour * 60 + opened.minute == 15 * 60)
    assert np.all(closed.hour * 60 + closed.minute <= 15 * 60 + 55)
    timed = trades["reason"] == core.EXIT_SIGNAL
    assert timed.mean() > 0.5
    assert np.all((closed.hour * 60 + closed.minute)[timed] == 15 * 60 + 55)
    assert set(trades["reason"]) <= {core.EXIT_SIGNAL, core.EXIT_STOP, core.EXIT_STOP_GAP}
    assert np.all(trades["stop_distance"] > 0)


def test_a_missing_entry_bar_cancels_the_trade_instead_of_entering_late() -> None:
    frame = frame_for_tests(4)
    params = ff.pre_fixing_params("USDJPY")
    london = frame.index.tz_convert("Europe/London")
    day = london[np.flatnonzero((london.hour == 15) & (london.minute == 0))[5]].date()
    missing = frame.index[(london.date == day) & (london.hour == 15) & (london.minute == 0)]
    gapped = frame.drop(missing)

    complete = simulate(
        ff.fixing_flow_step, ff.fixing_flow_init, params, STATE, fx.bars_to_matrix(frame),
        anchor_signal=True, max_entry_gap=300.0,
    )
    trades = simulate(
        ff.fixing_flow_step, ff.fixing_flow_init, params, STATE, fx.bars_to_matrix(gapped),
        anchor_signal=True, max_entry_gap=300.0,
    )

    assert len(missing) == 1
    assert len(trades) == len(complete) - 1
    opened = gapped.index[trades["entry_index"]].tz_convert("Europe/London")
    assert np.all(opened.hour * 60 + opened.minute == 15 * 60)


@pytest.mark.parametrize("builder", [ff.pre_fixing_params, ff.post_fixing_params], ids=["F2a", "F2b"])
def test_streaming_matches_batch_and_the_future_never_changes_the_past(builder) -> None:
    matrix = fx.bars_to_matrix(frame_for_tests(5))
    params = builder("USDCAD")

    assert_streaming_equals_batch(ff.fixing_flow_step, ff.fixing_flow_init, params, STATE, matrix)
    assert_the_future_never_changes_the_past(ff.fixing_flow_step, ff.fixing_flow_init, params, STATE, matrix)


def test_the_params_reject_nonsense() -> None:
    good = dict(pair_side=fx.LONG, entry_time="15:00", exit_time="15:55")
    assert ff.fixing_flow_params(**good).shape == (6,)
    for override in (
        dict(pair_side=0),
        dict(exit_time="14:00"),
        dict(zone=5),
        dict(stop_atr=0),
        dict(bar_seconds=45),
    ):
        with pytest.raises(ValueError):
            ff.fixing_flow_params(**{**good, **override})


def test_the_atr_counts_gaps_between_bars_and_ignores_the_spread() -> None:
    frame = synthetic_frame(start="2024-03-04", weeks=8, bar_seconds=300, seed=31, gap_sigma_pips=4.0, spread_jitter=True)
    matrix = fx.bars_to_matrix(frame)

    intents, stops, _ = decisions(ff.fixing_flow_step, ff.fixing_flow_init, ff.pre_fixing_params("USDJPY"), STATE, matrix)
    expected_index, expected_stop = expected_entries(frame, "15:00")

    fired = np.flatnonzero(intents != fx.HOLD)
    assert fired.tolist() == expected_index.tolist()
    assert np.allclose(stops[fired], expected_stop, rtol=0, atol=1e-12)


def test_the_post_fixing_leg_enters_at_16_05_and_leaves_at_17_00_london_time() -> None:
    frame = frame_for_tests(6)
    trades = simulate(ff.fixing_flow_step, ff.fixing_flow_init, ff.post_fixing_params("USDJPY"), STATE, fx.bars_to_matrix(frame))

    opened = frame.index[trades["entry_index"]].tz_convert("Europe/London")
    closed = frame.index[trades["exit_index"]].tz_convert("Europe/London")
    timed = trades["reason"] == core.EXIT_SIGNAL
    assert len(trades) >= 30 and np.all(opened.hour * 60 + opened.minute == 16 * 60 + 5)
    assert np.all((closed.hour * 60 + closed.minute)[timed] == 17 * 60)
