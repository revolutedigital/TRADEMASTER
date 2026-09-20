"""Negative controls (C1 mean reversion on M1, C2 momentum on M5): rules that should NOT pay.

Both are textbook rules with a tight stop and target, chosen because at retail costs they are
expected to lose. They exist to test the test: if the lab ever approves one of them, the suspect
is a defect in the data, the costs or the simulator, and not a discovery.

The exact rules are in docs/forex/fast-preregistration.md (C1 and C2).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from app.fx.sessions import NEW_YORK, local_time_of_day, seconds_of_day
from app.fx.strategies.common import ring_count, ring_mean, ring_push, ring_slots, ring_std
from app.fx.strategy import (
    ASK_CLOSE,
    BAR_TIME,
    BID_CLOSE,
    ENTER_LONG,
    ENTER_SHORT,
    EXIT,
    FLAT,
    HOLD,
)

BOLLINGER_WINDOW = 20
RETURN_WINDOW = 48

_BAND_WIDTH, _RETURN_Z = 0, 0  # the first parameter of each control
_STOP, _TARGET, _MAX_BARS, _BLACKOUT_START, _BLACKOUT_END, _BAR_SECONDS = 1, 2, 3, 4, 5, 6

_C1_CLOSES = 0
_C1_HELD = _C1_CLOSES + ring_slots(BOLLINGER_WINDOW)
BOLLINGER_REVERSION_STATE_SIZE = _C1_HELD + 1

_C2_PREVIOUS_CLOSE, _C2_HAS_PREVIOUS, _C2_HELD = 0, 1, 2
_C2_RETURNS = 3
MOMENTUM_BURST_STATE_SIZE = _C2_RETURNS + ring_slots(RETURN_WINDOW)


def _control_params(
    threshold: float, stop_pips: float, target_pips: float, pip_size: float, max_bars: int, bar_seconds: int
) -> np.ndarray:
    if threshold <= 0 or stop_pips <= 0 or target_pips <= 0 or pip_size <= 0 or max_bars < 1:
        raise ValueError("threshold, stop, target, pip size and max_bars must be positive")
    return np.array(
        [
            threshold, stop_pips * pip_size, target_pips * pip_size, max_bars,
            seconds_of_day("16:55"), seconds_of_day("17:15"), bar_seconds,
        ],
        dtype=np.float64,
    )


def bollinger_reversion_params(pip_size: float) -> np.ndarray:
    """C1: fade a close outside Bollinger(20, 2) on M1 bars, 3 pips stop and target, 30 bars."""
    return _control_params(2.0, 3.0, 3.0, pip_size, 30, 60)


def momentum_burst_params(pip_size: float) -> np.ndarray:
    """C2: follow an M5 return of at least 2.5 standard deviations, 3 pips stop and target, 12 bars."""
    return _control_params(2.5, 3.0, 3.0, pip_size, 12, 300)


@njit(cache=True)
def _blacked_out(params, bar):
    entry_time = float(local_time_of_day(bar[BAR_TIME] + params[_BAR_SECONDS], NEW_YORK))
    return params[_BLACKOUT_START] <= entry_time < params[_BLACKOUT_END]


@njit(cache=True)
def bollinger_reversion_init(params, state):
    for i in range(state.shape[0]):
        state[i] = 0.0


@njit(cache=True)
def bollinger_reversion_step(params, state, bar, position):
    mid_close = 0.5 * (bar[BID_CLOSE] + bar[ASK_CLOSE])
    ring_push(state, _C1_CLOSES, BOLLINGER_WINDOW, mid_close)  # the band includes the current close
    intent = HOLD
    stop_distance = 0.0
    target_distance = 0.0
    if position != FLAT:
        state[_C1_HELD] += 1.0
        if state[_C1_HELD] >= params[_MAX_BARS]:
            intent = EXIT
    else:
        state[_C1_HELD] = 0.0
        if ring_count(state, _C1_CLOSES) == BOLLINGER_WINDOW and not _blacked_out(params, bar):
            mean = ring_mean(state, _C1_CLOSES)
            width = params[_BAND_WIDTH] * ring_std(state, _C1_CLOSES)
            if width > 0.0:
                if mid_close > mean + width:
                    intent = ENTER_SHORT
                elif mid_close < mean - width:
                    intent = ENTER_LONG
                if intent != HOLD:
                    stop_distance = params[_STOP]
                    target_distance = params[_TARGET]
    return intent, stop_distance, target_distance


@njit(cache=True)
def momentum_burst_init(params, state):
    for i in range(state.shape[0]):
        state[i] = 0.0


@njit(cache=True)
def momentum_burst_step(params, state, bar, position):
    mid_close = 0.5 * (bar[BID_CLOSE] + bar[ASK_CLOSE])
    has_previous = state[_C2_HAS_PREVIOUS] > 0.5
    change = mid_close - state[_C2_PREVIOUS_CLOSE] if has_previous else 0.0
    intent = HOLD
    stop_distance = 0.0
    target_distance = 0.0
    if position != FLAT:
        state[_C2_HELD] += 1.0
        if state[_C2_HELD] >= params[_MAX_BARS]:
            intent = EXIT
    else:
        state[_C2_HELD] = 0.0
        if has_previous and ring_count(state, _C2_RETURNS) == RETURN_WINDOW:
            deviation = ring_std(state, _C2_RETURNS)  # the previous returns, not the current one
            if (
                deviation > 0.0
                and abs(change) >= params[_RETURN_Z] * deviation
                and not _blacked_out(params, bar)
            ):
                intent = ENTER_LONG if change > 0.0 else ENTER_SHORT
                stop_distance = params[_STOP]
                target_distance = params[_TARGET]
    if has_previous:
        ring_push(state, _C2_RETURNS, RETURN_WINDOW, change)
    state[_C2_PREVIOUS_CLOSE] = mid_close
    state[_C2_HAS_PREVIOUS] = 1.0
    return intent, stop_distance, target_distance
