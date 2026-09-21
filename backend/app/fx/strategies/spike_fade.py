"""Family 3: fade a volatility spike (F3a with a threshold of 4, F3b with 6).

A spike is an M5 bar whose range is several times the median range of the previous 48 bars, that
closed mostly in one direction, while the spread has not blown out (a wide spread means the
market is not really quoting, and the fill would be fictional). The bot trades against the spike:
the stop sits half a spike range beyond its extreme, the target is the retracement of half of the
spike range from that extreme, and a position that has done neither is closed after a fixed
number of bars. It never enters around the 17:00 New York rollover, when spreads are unreliable.

The exact rules are in docs/forex/fast-preregistration.md (F3a and F3b).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from app.fx.sessions import NEW_YORK, local_time_of_day, seconds_of_day
from app.fx.strategies.common import ring_count, ring_median, ring_push, ring_slots
from app.fx.strategy import (
    ASK_CLOSE,
    ASK_HIGH,
    ASK_LOW,
    ASK_OPEN,
    BAR_TIME,
    BID_CLOSE,
    BID_HIGH,
    BID_LOW,
    BID_OPEN,
    ENTER_LONG,
    ENTER_SHORT,
    EXIT,
    FLAT,
    HOLD,
)

LOOKBACK = 48

_THRESHOLD, _BODY_FRACTION, _SPREAD_MULTIPLE, _STOP_RANGE, _RETRACE = 0, 1, 2, 3, 4
_MAX_BARS, _BLACKOUT_START, _BLACKOUT_END, _BAR_SECONDS = 5, 6, 7, 8

_RANGES = 0
_SPREADS = _RANGES + ring_slots(LOOKBACK)
_HELD = _SPREADS + ring_slots(LOOKBACK)
SPIKE_FADE_STATE_SIZE = _HELD + 1


def spike_fade_params(
    *,
    threshold: float,
    body_fraction: float = 0.6,
    spread_multiple: float = 2.0,
    stop_range: float = 0.5,
    retrace: float = 0.5,
    max_bars: int = 12,
    blackout_start: str = "16:55",
    blackout_end: str = "17:15",
    bar_seconds: int = 300,
) -> np.ndarray:
    """Parameters for the fade; the blackout is in New York time."""
    if threshold <= 1 or spread_multiple <= 0 or stop_range <= 0 or max_bars < 1:
        raise ValueError("threshold must exceed 1 and the spread, stop and bar limits be positive")
    if not 0 < body_fraction <= 1 or not 0 < retrace < 1:
        raise ValueError("body_fraction must be in (0, 1] and retrace in (0, 1)")
    if bar_seconds < 60 or bar_seconds % 60:
        raise ValueError("bar_seconds must be a whole number of minutes")
    return np.array(
        [
            threshold, body_fraction, spread_multiple, stop_range, retrace, max_bars,
            seconds_of_day(blackout_start), seconds_of_day(blackout_end), bar_seconds,
        ],
        dtype=np.float64,
    )


def spike_fade_4_params() -> np.ndarray:
    """F3a."""
    return spike_fade_params(threshold=4.0)


def spike_fade_6_params() -> np.ndarray:
    """F3b."""
    return spike_fade_params(threshold=6.0)


@njit(cache=True)
def spike_fade_init(params, state):
    for i in range(state.shape[0]):
        state[i] = 0.0


@njit(cache=True)
def spike_fade_step(params, state, bar, position):
    mid_open = 0.5 * (bar[BID_OPEN] + bar[ASK_OPEN])
    mid_high = 0.5 * (bar[BID_HIGH] + bar[ASK_HIGH])
    mid_low = 0.5 * (bar[BID_LOW] + bar[ASK_LOW])
    mid_close = 0.5 * (bar[BID_CLOSE] + bar[ASK_CLOSE])
    bar_range = mid_high - mid_low
    body = mid_close - mid_open
    closing_spread = bar[ASK_CLOSE] - bar[BID_CLOSE]

    intent = HOLD
    stop_distance = 0.0
    target_distance = 0.0
    if position != FLAT:
        state[_HELD] += 1.0
        if state[_HELD] >= params[_MAX_BARS]:
            intent = EXIT
    else:
        state[_HELD] = 0.0
        if ring_count(state, _RANGES) == LOOKBACK:
            entry_time = float(local_time_of_day(bar[BAR_TIME] + params[_BAR_SECONDS], NEW_YORK))
            blacked_out = params[_BLACKOUT_START] <= entry_time < params[_BLACKOUT_END]
            # The same four conditions as always, cheapest first: a median sorts the ring, and most bars fail
            # the body test, so most bars never pay for one.
            is_spike = False
            if bar_range > 0.0 and abs(body) >= params[_BODY_FRACTION] * bar_range:
                is_spike = (
                    bar_range >= params[_THRESHOLD] * ring_median(state, _RANGES)
                    and closing_spread <= params[_SPREAD_MULTIPLE] * ring_median(state, _SPREADS)
                )
            if is_spike and not blacked_out:
                if body > 0.0:
                    stop_distance = mid_high + params[_STOP_RANGE] * bar_range - mid_close
                    target_distance = mid_close - (mid_high - params[_RETRACE] * bar_range)
                    candidate = ENTER_SHORT
                else:
                    stop_distance = mid_close - (mid_low - params[_STOP_RANGE] * bar_range)
                    target_distance = (mid_low + params[_RETRACE] * bar_range) - mid_close
                    candidate = ENTER_LONG
                if stop_distance > 0.0 and target_distance > 0.0:
                    intent = candidate
                else:
                    stop_distance = 0.0
                    target_distance = 0.0
    # The reference window must not contain the bar being judged, so it is updated last.
    ring_push(state, _RANGES, LOOKBACK, bar_range)
    ring_push(state, _SPREADS, LOOKBACK, closing_spread)
    return intent, stop_distance, target_distance
