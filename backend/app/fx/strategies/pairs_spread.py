"""Family 5: mean reversion of a cross that is the ratio of two highly correlated pairs (F5a, F5b).

EURGBP is exactly EURUSD divided by GBPUSD, and AUDNZD is AUDUSD divided by NZDUSD. Such a ratio
tends to oscillate in a band, and trading it directly is the same as trading the spread between
the two pairs with a single position: F5a trades the EURGBP quotes as they come, F5b builds the
AUDNZD ratio from the two AUDUSD and NZDUSD legs, paying both spreads (`synthetic_cross`).

On every H1 bar the z-score of the close against the mean and standard deviation of the previous
480 closes is computed. At |z| >= 2 the bot bets on the return to the mean: the target is the mean
at the moment of entry, the stop sits 3.5 standard deviations away on the far side, and a
position that has done neither is closed after 120 bars. No trade if |z| is already at the stop.

The exact rules are in docs/forex/fast-preregistration.md (F5a and F5b).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from app.fx.strategies.common import ring_count, ring_mean, ring_push, ring_slots, ring_std
from app.fx.strategy import (
    ASK_CLOSE,
    ASK_HIGH,
    ASK_LOW,
    ASK_OPEN,
    BAR_TIME,
    BAR_WIDTH,
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

WINDOW = 480

_ENTRY_Z, _STOP_Z, _MAX_BARS = 0, 1, 2

_CLOSES = 0
_HELD = _CLOSES + ring_slots(WINDOW)
PAIRS_SPREAD_STATE_SIZE = _HELD + 1


def pairs_spread_params(*, entry_z: float = 2.0, stop_z: float = 3.5, max_bars: int = 120) -> np.ndarray:
    if not 0 < entry_z < stop_z or max_bars < 1:
        raise ValueError("the thresholds need 0 < entry_z < stop_z and max_bars must be positive")
    return np.array([entry_z, stop_z, max_bars], dtype=np.float64)


@njit(cache=True)
def pairs_spread_init(params, state):
    for i in range(state.shape[0]):
        state[i] = 0.0


@njit(cache=True)
def pairs_spread_step(params, state, bar, position):
    mid_close = 0.5 * (bar[BID_CLOSE] + bar[ASK_CLOSE])
    intent = HOLD
    stop_distance = 0.0
    target_distance = 0.0
    if position != FLAT:
        state[_HELD] += 1.0
        if state[_HELD] >= params[_MAX_BARS]:
            intent = EXIT
    else:
        state[_HELD] = 0.0
        if ring_count(state, _CLOSES) == WINDOW:
            mean = ring_mean(state, _CLOSES)
            deviation = ring_std(state, _CLOSES)
            if deviation > 0.0:
                z = (mid_close - mean) / deviation
                if params[_ENTRY_Z] <= z < params[_STOP_Z]:
                    intent = ENTER_SHORT
                    stop_distance = mean + params[_STOP_Z] * deviation - mid_close
                    target_distance = mid_close - mean
                elif -params[_STOP_Z] < z <= -params[_ENTRY_Z]:
                    intent = ENTER_LONG
                    stop_distance = mid_close - (mean - params[_STOP_Z] * deviation)
                    target_distance = mean - mid_close
    # The window must not contain the bar being judged, so it is updated last.
    ring_push(state, _CLOSES, WINDOW, mid_close)
    return intent, stop_distance, target_distance


def synthetic_cross(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Minute bars of the ratio numerator / denominator, paying the spread of both legs.

    Buying the cross means buying the numerator pair at its ask and selling the denominator pair
    at its bid, so the cross ask is ask_num / bid_den and the cross bid is bid_num / ask_den. Only
    minutes that both legs quoted are kept. The ratio's high and low inside a minute are unknown
    (the two legs peak at different instants), so they are taken from the ratio at the minute's
    open and close, which never invents an extreme that the data did not show.
    """
    if numerator.shape[1] != BAR_WIDTH or denominator.shape[1] != BAR_WIDTH:
        raise ValueError("both legs must be (n, 9) bar matrices")
    common, num_index, den_index = np.intersect1d(
        numerator[:, BAR_TIME], denominator[:, BAR_TIME], return_indices=True
    )
    num, den = numerator[num_index], denominator[den_index]
    cross = np.empty((common.size, BAR_WIDTH), dtype=np.float64)
    cross[:, BAR_TIME] = common
    bid_open = num[:, BID_OPEN] / den[:, ASK_OPEN]
    bid_close = num[:, BID_CLOSE] / den[:, ASK_CLOSE]
    ask_open = num[:, ASK_OPEN] / den[:, BID_OPEN]
    ask_close = num[:, ASK_CLOSE] / den[:, BID_CLOSE]
    cross[:, BID_OPEN], cross[:, BID_CLOSE] = bid_open, bid_close
    cross[:, ASK_OPEN], cross[:, ASK_CLOSE] = ask_open, ask_close
    cross[:, BID_HIGH], cross[:, BID_LOW] = np.maximum(bid_open, bid_close), np.minimum(bid_open, bid_close)
    cross[:, ASK_HIGH], cross[:, ASK_LOW] = np.maximum(ask_open, ask_close), np.minimum(ask_open, ask_close)
    return cross
