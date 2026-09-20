"""Builds bid/ask bars from a live quote stream, the way the lab builds them from minutes.

A bar opens with the first quote of its epoch-aligned bucket and carries the open, high, low and
close of the bid and of the ask. It is emitted when a quote from a later bucket arrives, or when
the clock passes the end of its bucket (`close_due`). A bucket in which no quote arrives produces
no bar, exactly like a hole in the historical data.
"""

from __future__ import annotations

import numpy as np

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


class BarBuilder:
    def __init__(self, seconds: int) -> None:
        if seconds < 60 or seconds % 60:
            raise ValueError("a bar must be a whole number of minutes")
        self.seconds = seconds
        self._bar: np.ndarray | None = None

    def _bucket(self, time: float) -> float:
        return float(np.floor(time / self.seconds) * self.seconds)

    def on_quote(self, time: float, bid: float, ask: float) -> np.ndarray | None:
        """Add a quote; returns the bar that this quote closed, if any."""
        if ask < bid:
            raise ValueError("a crossed quote cannot be part of a bar")
        closed = None
        bucket = self._bucket(time)
        if self._bar is not None and bucket < self._bar[BAR_TIME]:
            raise ValueError("quotes must arrive in time order")
        if self._bar is not None and bucket > self._bar[BAR_TIME]:
            closed, self._bar = self._bar, None
        if self._bar is None:
            bar = np.empty(BAR_WIDTH)
            bar[BAR_TIME] = bucket
            bar[[BID_OPEN, BID_HIGH, BID_LOW, BID_CLOSE]] = bid
            bar[[ASK_OPEN, ASK_HIGH, ASK_LOW, ASK_CLOSE]] = ask
            self._bar = bar
        else:
            self._bar[BID_HIGH] = max(self._bar[BID_HIGH], bid)
            self._bar[BID_LOW] = min(self._bar[BID_LOW], bid)
            self._bar[BID_CLOSE] = bid
            self._bar[ASK_HIGH] = max(self._bar[ASK_HIGH], ask)
            self._bar[ASK_LOW] = min(self._bar[ASK_LOW], ask)
            self._bar[ASK_CLOSE] = ask
        return closed

    def close_due(self, now: float) -> np.ndarray | None:
        """Emit the open bar if the clock has passed the end of its bucket."""
        if self._bar is not None and now >= self._bar[BAR_TIME] + self.seconds:
            closed, self._bar = self._bar, None
            return closed
        return None
