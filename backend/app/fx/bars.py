"""Aggregate one-minute bid/ask bars into the coarser bars the strategies trade on.

Every strategy of the fast families works on M5, M15 or H1 bars, but the data is stored as
minutes. A coarse bar takes the open of its first minute, the highest high, the lowest low and
the close of its last minute, for the bid and for the ask separately, and is stamped with the
epoch-aligned instant at which it opens. A bucket with no minutes produces no bar, so holes in
the data stay holes instead of turning into flat bars.

The bucket boundaries depend only on the timestamps, so `bucket_starts` is computed once per pair
and reused to aggregate any number of price paths that share those timestamps (the placebo
datasets of the calibration).
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
    BAR_WIDTH,
    BID_CLOSE,
    BID_HIGH,
    BID_LOW,
    BID_OPEN,
)


def bucket_starts(times: np.ndarray, seconds: int) -> np.ndarray:
    """Index of the first minute of every non-empty `seconds`-long bucket aligned to the epoch."""
    if seconds < 60 or seconds % 60:
        raise ValueError("a bar must be a whole number of minutes")
    if times.size == 0:
        return np.empty(0, dtype=np.int64)
    if np.any(np.diff(times) <= 0):
        raise ValueError("bar times must be strictly increasing")
    buckets = np.floor_divide(times, seconds)
    return np.concatenate(([0], np.flatnonzero(np.diff(buckets)) + 1)).astype(np.int64)


@njit(cache=True)
def aggregate(matrix, starts, seconds):
    """Collapse `matrix` (n, 9) into one bar per bucket that begins at each index in `starts`."""
    count = starts.shape[0]
    bars = np.empty((count, BAR_WIDTH), dtype=np.float64)
    for bucket in range(count):
        first = starts[bucket]
        last = starts[bucket + 1] if bucket + 1 < count else matrix.shape[0]
        bars[bucket, BAR_TIME] = np.floor(matrix[first, BAR_TIME] / seconds) * seconds
        bars[bucket, BID_OPEN] = matrix[first, BID_OPEN]
        bars[bucket, ASK_OPEN] = matrix[first, ASK_OPEN]
        bars[bucket, BID_CLOSE] = matrix[last - 1, BID_CLOSE]
        bars[bucket, ASK_CLOSE] = matrix[last - 1, ASK_CLOSE]
        bid_high = matrix[first, BID_HIGH]
        ask_high = matrix[first, ASK_HIGH]
        bid_low = matrix[first, BID_LOW]
        ask_low = matrix[first, ASK_LOW]
        for row in range(first + 1, last):
            bid_high = max(bid_high, matrix[row, BID_HIGH])
            ask_high = max(ask_high, matrix[row, ASK_HIGH])
            bid_low = min(bid_low, matrix[row, BID_LOW])
            ask_low = min(ask_low, matrix[row, ASK_LOW])
        bars[bucket, BID_HIGH] = bid_high
        bars[bucket, ASK_HIGH] = ask_high
        bars[bucket, BID_LOW] = bid_low
        bars[bucket, ASK_LOW] = ask_low
    return bars


def aggregate_minutes(matrix: np.ndarray, seconds: int) -> np.ndarray:
    """Aggregate a minute matrix into `seconds`-long bars."""
    if seconds == 60:
        return np.ascontiguousarray(matrix)
    return aggregate(matrix, bucket_starts(matrix[:, BAR_TIME], seconds), seconds)
