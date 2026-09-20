"""Family 1: breakout of the range formed before a session opens (F1a London, F1b New York).

The hypothesis is that liquidity and volatility rise when a session opens and that the price tends
to keep going after it leaves the range built during the quiet hours before. Per local calendar
day, the range is the high and low of the mid over the range window; the first bar that closes
outside it during the signal window is the entry, in the direction of the break. Only one trade
per day. The trade is skipped when the range is unusually narrow or wide against the daily ATR
(the mean true range of the last 14 completed FX days), the stop is the opposite edge of the
range, the target is a fixed multiple of the stop, and any open position is closed at the exit time.

The exact rules are in docs/forex/fast-preregistration.md (F1a and F1b); every parameter below
is fixed there and is not to be tuned.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from app.fx.sessions import (
    LONDON,
    NEW_YORK,
    fx_day,
    local_day,
    local_time_of_day,
    seconds_of_day,
)
from app.fx.strategies.common import (
    ring_count,
    ring_mean,
    ring_push,
    ring_slots,
    true_range,
)
from app.fx.strategy import (
    ASK_CLOSE,
    ASK_HIGH,
    ASK_LOW,
    BAR_TIME,
    BID_CLOSE,
    BID_HIGH,
    BID_LOW,
    ENTER_LONG,
    ENTER_SHORT,
    EXIT,
    FLAT,
    HOLD,
)

ATR_DAYS = 14
NO_DAY = -1.0e9
UNSET_HIGH = -1.0e18
UNSET_LOW = 1.0e18

_ZONE, _RANGE_START, _RANGE_END, _SIGNAL_START, _SIGNAL_END, _EXIT_TIME = 0, 1, 2, 3, 4, 5
_BAR_SECONDS, _MIN_RANGE_BARS, _MIN_ATR, _MAX_ATR, _REWARD_RISK = 6, 7, 8, 9, 10

_LOCAL_DAY, _RANGE_HIGH, _RANGE_LOW, _RANGE_BARS, _TRADED, _ENTRY_DAY = 0, 1, 2, 3, 4, 5
_FX_DAY, _DAY_HIGH, _DAY_LOW, _DAY_CLOSE, _PREVIOUS_DAY_CLOSE, _HAS_PREVIOUS_DAY = 6, 7, 8, 9, 10, 11
_ATR_RING = 12
SESSION_BREAKOUT_STATE_SIZE = _ATR_RING + ring_slots(ATR_DAYS)


def session_breakout_params(
    *,
    zone: int,
    range_start: str,
    range_end: str,
    signal_start: str,
    signal_end: str,
    exit_time: str,
    bar_seconds: int = 900,
    min_range_bars: int = 24,
    min_atr_multiple: float = 0.3,
    max_atr_multiple: float = 1.2,
    reward_risk: float = 1.5,
) -> np.ndarray:
    """Parameters for the breakout; the times are local wall-clock times of `zone`."""
    if zone not in (LONDON, NEW_YORK):
        raise ValueError("zone must be sessions.LONDON or sessions.NEW_YORK")
    times = [seconds_of_day(value) for value in (range_start, range_end, signal_start, signal_end, exit_time)]
    if not times[0] < times[1] <= times[2] < times[3] <= times[4]:
        raise ValueError("the windows must run in order: range, then signal, then exit")
    if bar_seconds < 60 or bar_seconds % 60 or min_range_bars < 1 or reward_risk <= 0:
        raise ValueError("bar_seconds, min_range_bars and reward_risk must be positive")
    if not 0 < min_atr_multiple < max_atr_multiple:
        raise ValueError("the range filter needs 0 < min_atr_multiple < max_atr_multiple")
    return np.array(
        [zone, *times, bar_seconds, min_range_bars, min_atr_multiple, max_atr_multiple, reward_risk],
        dtype=np.float64,
    )


def london_open_breakout_params() -> np.ndarray:
    """F1a: the 00:00-08:00 London range, broken between 08:00 and 11:00, flat by 16:30."""
    return session_breakout_params(
        zone=LONDON, range_start="00:00", range_end="08:00",
        signal_start="08:00", signal_end="11:00", exit_time="16:30",
    )


def new_york_open_breakout_params() -> np.ndarray:
    """F1b: the 03:00-08:00 New York range, broken between 08:00 and 11:00, flat by 16:00.

    The window has 20 M15 bars, so the same 75% completeness as F1a means 15 bars (amendment of
    the pre-registration, registered before any result).
    """
    return session_breakout_params(
        zone=NEW_YORK, range_start="03:00", range_end="08:00",
        signal_start="08:00", signal_end="11:00", exit_time="16:00", min_range_bars=15,
    )


@njit(cache=True)
def session_breakout_init(params, state):
    for i in range(state.shape[0]):
        state[i] = 0.0
    state[_LOCAL_DAY] = NO_DAY
    state[_FX_DAY] = NO_DAY
    state[_RANGE_HIGH] = UNSET_HIGH
    state[_RANGE_LOW] = UNSET_LOW
    state[_ENTRY_DAY] = NO_DAY


@njit(cache=True)
def session_breakout_step(params, state, bar, position):
    zone = int(params[_ZONE])
    range_start = params[_RANGE_START]
    range_end = params[_RANGE_END]
    signal_start = params[_SIGNAL_START]
    signal_end = params[_SIGNAL_END]
    exit_time = params[_EXIT_TIME]
    bar_seconds = params[_BAR_SECONDS]

    opened = bar[BAR_TIME]
    mid_high = 0.5 * (bar[BID_HIGH] + bar[ASK_HIGH])
    mid_low = 0.5 * (bar[BID_LOW] + bar[ASK_LOW])
    mid_close = 0.5 * (bar[BID_CLOSE] + bar[ASK_CLOSE])

    # The daily ATR is built from FX days that are already complete when a new one begins.
    current_fx_day = float(fx_day(opened))
    if current_fx_day != state[_FX_DAY]:
        if state[_FX_DAY] != NO_DAY:
            ring_push(
                state, _ATR_RING, ATR_DAYS,
                true_range(
                    state[_DAY_HIGH], state[_DAY_LOW], state[_PREVIOUS_DAY_CLOSE],
                    state[_HAS_PREVIOUS_DAY] > 0.5,
                ),
            )
            state[_PREVIOUS_DAY_CLOSE] = state[_DAY_CLOSE]
            state[_HAS_PREVIOUS_DAY] = 1.0
        state[_FX_DAY] = current_fx_day
        state[_DAY_HIGH] = mid_high
        state[_DAY_LOW] = mid_low
    else:
        state[_DAY_HIGH] = max(state[_DAY_HIGH], mid_high)
        state[_DAY_LOW] = min(state[_DAY_LOW], mid_low)
    state[_DAY_CLOSE] = mid_close

    # A new local calendar day starts a fresh range and re-arms the one trade per day.
    today = float(local_day(opened, zone))
    if today != state[_LOCAL_DAY]:
        state[_LOCAL_DAY] = today
        state[_RANGE_HIGH] = UNSET_HIGH
        state[_RANGE_LOW] = UNSET_LOW
        state[_RANGE_BARS] = 0.0
        state[_TRADED] = 0.0
    time_of_day = float(local_time_of_day(opened, zone))
    if range_start <= time_of_day < range_end:
        state[_RANGE_HIGH] = max(state[_RANGE_HIGH], mid_high)
        state[_RANGE_LOW] = min(state[_RANGE_LOW], mid_low)
        state[_RANGE_BARS] += 1.0

    intent = HOLD
    stop_distance = 0.0
    target_distance = 0.0
    if position != FLAT:
        if time_of_day + bar_seconds >= exit_time or today != state[_ENTRY_DAY]:
            intent = EXIT
    elif (
        signal_start <= time_of_day < signal_end
        and state[_TRADED] < 0.5
        and state[_RANGE_BARS] >= params[_MIN_RANGE_BARS]
        and ring_count(state, _ATR_RING) == ATR_DAYS
    ):
        range_high = state[_RANGE_HIGH]
        range_low = state[_RANGE_LOW]
        atr = ring_mean(state, _ATR_RING)
        width = range_high - range_low
        if params[_MIN_ATR] * atr <= width <= params[_MAX_ATR] * atr:
            if mid_close > range_high:
                intent = ENTER_LONG
                stop_distance = mid_close - range_low
            elif mid_close < range_low:
                intent = ENTER_SHORT
                stop_distance = range_high - mid_close
            if intent != HOLD:
                target_distance = params[_REWARD_RISK] * stop_distance
                state[_TRADED] = 1.0
                state[_ENTRY_DAY] = today
    return intent, stop_distance, target_distance
