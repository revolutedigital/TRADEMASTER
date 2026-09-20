"""F1: the session breakout equals an independent pandas implementation of the pre-registered rule."""

import numpy as np
import pandas as pd
import pytest

from app.fx import sessions
from app.fx import strategy as fx
from app.fx.sim import core
from app.fx.strategies import session_breakout as sb
from tests.unit.fx_strategy_checks import (
    assert_streaming_equals_batch,
    assert_the_future_never_changes_the_past,
    decisions,
    mid_frame,
    simulate,
    synthetic_frame,
)

STATE = sb.SESSION_BREAKOUT_STATE_SIZE
LONDON_CASE = dict(
    zone_name="Europe/London", range_start="00:00", range_end="08:00", signal_start="08:00",
    signal_end="11:00", min_bars=24,
)
NEW_YORK_CASE = dict(
    zone_name="America/New_York", range_start="03:00", range_end="08:00", signal_start="08:00",
    signal_end="11:00", min_bars=15,
)


def frame_for_tests(seed: int = 1) -> pd.DataFrame:
    # 2024-02-26 to 2024-04-22 crosses the US switch (Mar 10) and the European one (Mar 31).
    return synthetic_frame(start="2024-02-26", weeks=8, bar_seconds=900, seed=seed)


def expected_breakouts(
    frame: pd.DataFrame, *, zone_name, range_start, range_end, signal_start, signal_end, min_bars
):
    """The rule, re-implemented from the pre-registration with pandas and no shared code."""
    mid = mid_frame(frame)
    local = mid.index.tz_convert(zone_name)
    clock = local.hour * 3600 + local.minute * 60
    wall_day = pd.Series(local.tz_localize(None).floor("D"), index=mid.index)
    new_york_wall = mid.index.tz_convert("America/New_York").tz_localize(None) - pd.Timedelta(hours=17)
    fx_label = pd.Series(new_york_wall.floor("D"), index=mid.index)

    daily = mid.groupby(fx_label).agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    previous_close = daily["close"].shift(1)
    true_range = pd.concat(
        [daily["high"] - daily["low"], (daily["high"] - previous_close).abs(), (daily["low"] - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(14).mean().shift(1)

    def seconds(text: str) -> int:
        return sessions.seconds_of_day(text)

    found = []
    for _, day_bars in mid.groupby(wall_day):
        index = day_bars.index
        in_range = (clock[mid.index.get_indexer(index)] >= seconds(range_start)) & (
            clock[mid.index.get_indexer(index)] < seconds(range_end)
        )
        if in_range.sum() < min_bars:
            continue
        high, low = day_bars["high"][in_range].max(), day_bars["low"][in_range].min()
        in_signal = (clock[mid.index.get_indexer(index)] >= seconds(signal_start)) & (
            clock[mid.index.get_indexer(index)] < seconds(signal_end)
        )
        for stamp in index[in_signal]:
            day_atr = atr.get(fx_label[stamp])
            if day_atr is None or np.isnan(day_atr) or not 0.3 * day_atr <= high - low <= 1.2 * day_atr:
                break
            close = day_bars.at[stamp, "close"]
            if close > high:
                found.append((mid.index.get_loc(stamp), fx.ENTER_LONG, close - low))
                break
            if close < low:
                found.append((mid.index.get_loc(stamp), fx.ENTER_SHORT, high - close))
                break
    return found


@pytest.mark.parametrize(
    ("params", "case"),
    [(sb.london_open_breakout_params(), LONDON_CASE), (sb.new_york_open_breakout_params(), NEW_YORK_CASE)],
    ids=["F1a-london", "F1b-new-york"],
)
@pytest.mark.parametrize("seed", [1, 2])
def test_the_signals_equal_the_independent_pandas_rule(params, case, seed) -> None:
    frame = frame_for_tests(seed)
    matrix = fx.bars_to_matrix(frame)

    intents, stops, targets = decisions(sb.session_breakout_step, sb.session_breakout_init, params, STATE, matrix)
    expected = expected_breakouts(frame, **case)

    fired = np.flatnonzero(intents != fx.HOLD)
    assert len(expected) >= 8
    assert fired.tolist() == [index for index, _, _ in expected]
    assert intents[fired].tolist() == [side for _, side, _ in expected]
    assert np.allclose(stops[fired], [distance for _, _, distance in expected], rtol=0, atol=1e-12)
    assert np.allclose(targets[fired], 1.5 * stops[fired], rtol=0, atol=1e-12)


def test_the_range_width_filter_removes_some_days_but_not_all() -> None:
    frame = frame_for_tests(3)
    params = sb.london_open_breakout_params()
    wide = sb.session_breakout_params(
        zone=sessions.LONDON, range_start="00:00", range_end="08:00", signal_start="08:00",
        signal_end="11:00", exit_time="16:30", min_atr_multiple=1e-6, max_atr_multiple=1e6,
    )
    matrix = fx.bars_to_matrix(frame)

    filtered = decisions(sb.session_breakout_step, sb.session_breakout_init, params, STATE, matrix)[0]
    unfiltered = decisions(sb.session_breakout_step, sb.session_breakout_init, wide, STATE, matrix)[0]

    assert 0 < (filtered != fx.HOLD).sum() < (unfiltered != fx.HOLD).sum()


def test_a_day_with_too_few_range_bars_is_skipped() -> None:
    frame = frame_for_tests(4)
    params = sb.london_open_breakout_params()
    matrix = fx.bars_to_matrix(frame)
    fired = np.flatnonzero(
        decisions(sb.session_breakout_step, sb.session_breakout_init, params, STATE, matrix)[0] != fx.HOLD
    )
    victim_day = frame.index[fired[len(fired) // 2]].tz_convert("Europe/London").date()
    local = frame.index.tz_convert("Europe/London")
    in_range = (local.date == victim_day) & (local.hour < 8)
    holes = np.flatnonzero(in_range)[8:20]  # 12 of the 32 range bars go missing
    gapped = frame.drop(frame.index[holes])

    after = decisions(
        sb.session_breakout_step, sb.session_breakout_init, params, STATE, fx.bars_to_matrix(gapped)
    )[0]
    fired_days = [gapped.index[i].tz_convert("Europe/London").date() for i in np.flatnonzero(after != fx.HOLD)]

    assert victim_day not in fired_days
    assert len(fired_days) == len(fired) - 1


def test_there_is_at_most_one_entry_per_local_day_and_only_inside_the_signal_window() -> None:
    frame = frame_for_tests(5)
    matrix = fx.bars_to_matrix(frame)
    intents = decisions(
        sb.session_breakout_step, sb.session_breakout_init, sb.london_open_breakout_params(), STATE, matrix
    )[0]

    fired = frame.index[intents != fx.HOLD].tz_convert("Europe/London")

    assert len(set(fired.date)) == len(fired)
    minutes = fired.hour * 60 + fired.minute
    assert np.all((minutes >= 8 * 60) & (minutes < 11 * 60))


def test_trades_close_at_the_exit_time_of_the_same_day_and_carry_the_range_stop() -> None:
    frame = frame_for_tests(6)
    matrix = fx.bars_to_matrix(frame)
    params = sb.london_open_breakout_params()

    trades = simulate(sb.session_breakout_step, sb.session_breakout_init, params, STATE, matrix)
    expected = expected_breakouts(frame, **LONDON_CASE)

    assert len(trades) == len(expected) >= 8
    opened = frame.index[trades["entry_index"]].tz_convert("Europe/London")
    closed = frame.index[trades["exit_index"]].tz_convert("Europe/London")
    assert np.array_equal(trades["entry_index"], [index + 1 for index, _, _ in expected])
    assert np.allclose(trades["stop_distance"], [d for _, _, d in expected], rtol=0, atol=1e-12)
    assert np.array_equal(opened.date, closed.date)
    assert np.all(closed.hour * 60 + closed.minute <= 16 * 60 + 30)
    signal_exits = trades["reason"] == core.EXIT_SIGNAL
    assert signal_exits.any()
    assert np.all((closed.hour * 60 + closed.minute)[signal_exits] == 16 * 60 + 30)


def test_it_stays_out_while_holding_a_position_and_exits_at_the_exit_time() -> None:
    frame = frame_for_tests(7)
    matrix = fx.bars_to_matrix(frame)
    runner = fx.StreamingRunner(
        sb.session_breakout_step, sb.session_breakout_init, sb.london_open_breakout_params(), STATE
    )

    position, entries, exits, entered_while_holding = fx.FLAT, [], [], 0
    for index, bar in enumerate(matrix):
        intent = runner.on_bar(bar, position)[0]
        if intent in (fx.ENTER_LONG, fx.ENTER_SHORT):
            entered_while_holding += position != fx.FLAT
            entries.append(index)
            position = fx.LONG  # the order fills at the next open, so the next bar sees the position
        elif intent == fx.EXIT:
            exits.append(index)
            position = fx.FLAT

    assert entries and entered_while_holding == 0
    assert len(exits) == len(entries)
    closes = frame.index[exits].tz_convert("Europe/London") + pd.Timedelta(minutes=15)
    assert np.all(closes.hour * 60 + closes.minute == 16 * 60 + 30)


def test_the_new_york_range_uses_new_york_time_not_london_time() -> None:
    frame = frame_for_tests(8)
    matrix = fx.bars_to_matrix(frame)
    intents = decisions(
        sb.session_breakout_step, sb.session_breakout_init, sb.new_york_open_breakout_params(), STATE, matrix
    )[0]

    fired = frame.index[intents != fx.HOLD].tz_convert("America/New_York")

    assert len(fired) > 0
    assert np.all((fired.hour * 60 + fired.minute >= 8 * 60) & (fired.hour * 60 + fired.minute < 11 * 60))


@pytest.mark.parametrize(
    "params", [sb.london_open_breakout_params(), sb.new_york_open_breakout_params()], ids=["F1a", "F1b"]
)
def test_streaming_matches_batch_and_the_future_never_changes_the_past(params) -> None:
    matrix = fx.bars_to_matrix(frame_for_tests(9))

    assert_streaming_equals_batch(sb.session_breakout_step, sb.session_breakout_init, params, STATE, matrix)
    assert_the_future_never_changes_the_past(
        sb.session_breakout_step, sb.session_breakout_init, params, STATE, matrix
    )


def test_the_params_reject_nonsense() -> None:
    good = dict(
        zone=sessions.LONDON, range_start="00:00", range_end="08:00", signal_start="08:00",
        signal_end="11:00", exit_time="16:30",
    )
    assert sb.session_breakout_params(**good).shape == (11,)
    for override in (
        dict(zone=7),
        dict(range_end="07:00", signal_start="06:00"),
        dict(signal_end="17:00"),
        dict(reward_risk=0),
        dict(min_atr_multiple=2.0),
        dict(bar_seconds=90),
        dict(min_range_bars=0),
    ):
        with pytest.raises(ValueError):
            sb.session_breakout_params(**{**good, **override})


def test_the_stop_level_is_the_opposite_edge_of_the_range_when_anchored_to_the_signal() -> None:
    frame = frame_for_tests(11)
    matrix = fx.bars_to_matrix(frame)
    mid = mid_frame(frame)
    london = mid.index.tz_convert("Europe/London")

    trades = simulate(
        sb.session_breakout_step, sb.session_breakout_init, sb.london_open_breakout_params(), STATE, matrix,
        anchor_signal=True, max_entry_gap=900.0,
    )

    assert len(trades) >= 8
    for _, trade in trades.iterrows():
        day = london[int(trade["entry_index"])].date()
        window = mid[(london.date == day) & (london.hour < 8)]
        edge = window["low"].min() if trade["side"] == fx.LONG else window["high"].max()
        stop_level = trade["entry_price"] - trade["side"] * trade["stop_distance"]
        assert stop_level == pytest.approx(edge, abs=1e-9)


@pytest.mark.parametrize(
    ("params", "case"),
    [(sb.london_open_breakout_params(), LONDON_CASE), (sb.new_york_open_breakout_params(), NEW_YORK_CASE)],
    ids=["F1a-london", "F1b-new-york"],
)
def test_the_signals_equal_the_pandas_rule_with_gaps_between_bars_and_variable_spreads(params, case) -> None:
    frame = synthetic_frame(start="2024-02-26", weeks=8, bar_seconds=900, seed=21, gap_sigma_pips=4.0, spread_jitter=True)
    matrix = fx.bars_to_matrix(frame)

    intents, stops, _ = decisions(sb.session_breakout_step, sb.session_breakout_init, params, STATE, matrix)
    expected = expected_breakouts(frame, **case)

    fired = np.flatnonzero(intents != fx.HOLD)
    assert len(expected) >= 6
    assert fired.tolist() == [index for index, _, _ in expected]
    assert np.allclose(stops[fired], [distance for _, _, distance in expected], rtol=0, atol=1e-12)


@pytest.mark.parametrize(("zone", "params", "range_bars"), [
    ("Europe/London", sb.london_open_breakout_params(), 24), ("America/New_York", sb.new_york_open_breakout_params(), 15),
], ids=["F1a", "F1b"])
def test_the_range_needs_exactly_the_minimum_number_of_bars(zone, params, range_bars) -> None:
    frame = frame_for_tests(4)
    matrix = fx.bars_to_matrix(frame)
    fired = np.flatnonzero(decisions(sb.session_breakout_step, sb.session_breakout_init, params, STATE, matrix)[0] != fx.HOLD)
    victim = frame.index[fired[len(fired) // 2]].tz_convert(zone).date()
    local = frame.index.tz_convert(zone)
    first_hour = 0 if zone == "Europe/London" else 3
    in_range = np.flatnonzero((local.date == victim) & (local.hour >= first_hour) & (local.hour < 8))
    window = len(in_range)
    mid = mid_frame(frame).iloc[in_range]
    # never drop the bars that set the range edges, so the range itself does not change
    edges = {int(np.argmax(mid["high"].to_numpy())), int(np.argmin(mid["low"].to_numpy()))}
    droppable = [position for i, position in enumerate(in_range) if i not in edges]

    def fires_on_victim(kept: int) -> bool:
        gapped = frame.drop(frame.index[droppable[: window - kept]])
        after = decisions(sb.session_breakout_step, sb.session_breakout_init, params, STATE, fx.bars_to_matrix(gapped))[0]
        return victim in {gapped.index[i].tz_convert(zone).date() for i in np.flatnonzero(after != fx.HOLD)}

    assert fires_on_victim(range_bars) is True
    assert fires_on_victim(range_bars - 1) is False


def test_the_new_york_variant_leaves_at_sixteen_hundred_new_york_time() -> None:
    frame = frame_for_tests(12)
    trades = simulate(
        sb.session_breakout_step, sb.session_breakout_init, sb.new_york_open_breakout_params(), STATE,
        fx.bars_to_matrix(frame),
    )

    timed = trades[trades["reason"] == core.EXIT_SIGNAL]
    closed = frame.index[timed["exit_index"]].tz_convert("America/New_York")
    assert len(timed) > 0 and np.all(closed.hour * 60 + closed.minute == 16 * 60)
