"""Placebo data with no edge by construction, for key B of the pre-registered criterion.

Each minute, the move of the mid price away from the previous close changes sign with probability
one half, using one coin per minute shared by every pair (which keeps the correlation between
pairs and the coherence of the crosses). The high and low of a mirrored minute swap roles, and the
spreads of the original minute stay where they were. Nothing predictable is left in the price
(no trend, no reversal, no drift by hour of day), while the volatility by hour, its clustering, the
spreads and the holes of the real data are kept.

Nothing here touches the trading engine, the database, or an exchange.
"""

from __future__ import annotations

import numpy as np
from numba import njit

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
)

SECONDS_PER_MINUTE = 60


def coin_table(first_minute: int, last_minute: int, seed: int) -> np.ndarray:
    """One +1/-1 coin for every minute number (seconds since the epoch / 60) in the range."""
    if last_minute < first_minute:
        raise ValueError("last_minute must not precede first_minute")
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, last_minute - first_minute + 1).astype(np.int8) * 2 - 1


def coins_for(matrix: np.ndarray, table: np.ndarray, first_minute: int) -> np.ndarray:
    minutes = (matrix[:, BAR_TIME] // SECONDS_PER_MINUTE).astype(np.int64) - first_minute
    if minutes.size and (minutes.min() < 0 or minutes.max() >= table.size):
        raise ValueError("the bars fall outside the coin table")
    return table[minutes]


@njit(cache=True)
def sign_randomize(matrix, coins):
    """Rebuild the minute bars with the coin's sign on every move away from the previous close."""
    count = matrix.shape[0]
    out = np.empty_like(matrix)
    previous_original = 0.5 * (matrix[0, BID_OPEN] + matrix[0, ASK_OPEN])
    previous_new = previous_original
    for t in range(count):
        sign = float(coins[t])
        mid_open = 0.5 * (matrix[t, BID_OPEN] + matrix[t, ASK_OPEN])
        mid_high = 0.5 * (matrix[t, BID_HIGH] + matrix[t, ASK_HIGH])
        mid_low = 0.5 * (matrix[t, BID_LOW] + matrix[t, ASK_LOW])
        mid_close = 0.5 * (matrix[t, BID_CLOSE] + matrix[t, ASK_CLOSE])
        half_open = 0.5 * (matrix[t, ASK_OPEN] - matrix[t, BID_OPEN])
        half_high = 0.5 * (matrix[t, ASK_HIGH] - matrix[t, BID_HIGH])
        half_low = 0.5 * (matrix[t, ASK_LOW] - matrix[t, BID_LOW])
        half_close = 0.5 * (matrix[t, ASK_CLOSE] - matrix[t, BID_CLOSE])

        new_open = previous_new + sign * (mid_open - previous_original)
        new_close = previous_new + sign * (mid_close - previous_original)
        if sign > 0:
            new_high = previous_new + (mid_high - previous_original)
            new_low = previous_new + (mid_low - previous_original)
            half_new_high, half_new_low = half_high, half_low
        else:
            new_high = previous_new - (mid_low - previous_original)
            new_low = previous_new - (mid_high - previous_original)
            half_new_high, half_new_low = half_low, half_high

        bid_open, ask_open = new_open - half_open, new_open + half_open
        bid_close, ask_close = new_close - half_close, new_close + half_close
        out[t, BAR_TIME] = matrix[t, BAR_TIME]
        out[t, BID_OPEN], out[t, ASK_OPEN] = bid_open, ask_open
        out[t, BID_CLOSE], out[t, ASK_CLOSE] = bid_close, ask_close
        out[t, BID_HIGH] = max(new_high - half_new_high, bid_open, bid_close)
        out[t, ASK_HIGH] = max(new_high + half_new_high, ask_open, ask_close)
        out[t, BID_LOW] = min(new_low - half_new_low, bid_open, bid_close)
        out[t, ASK_LOW] = min(new_low + half_new_low, ask_open, ask_close)
        previous_original = mid_close
        previous_new = new_close
    return out
