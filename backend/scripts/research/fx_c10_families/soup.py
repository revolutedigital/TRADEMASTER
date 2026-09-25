"""Turtle Soup (Linda Raschke and Laurence Connors, "Street Smarts", 1995): fade a false breakout. When price trades
below the lowest low of the previous N completed days (for N = 20, a low set at least four days earlier) and a bar
then closes back above that level, buy; mirrored for highs. The day's extreme so far is the structural stop. N = 1
is the same fade of the previous day's high and low. One signal a day."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.research import fx_c10_core as core

PARAMS = [{"timeframe": "M15", "lookback": 20}, {"timeframe": "M15", "lookback": 1}]
MIN_AGE_DAYS = 4


def levels(d1: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """For every position j (the number of full days before a bar's day, 0 to len(d1)), the lowest low and highest
    high of full days j-lookback .. j-1 (NaN when not usable)."""
    n = len(d1) + 1
    low, high = np.full(n, np.nan), np.full(n, np.nan)
    for j in range(lookback, n):
        lows, highs = d1[j - lookback:j, 2], d1[j - lookback:j, 1]
        a, b = int(np.argmin(lows)), int(np.argmax(highs))
        if lookback == 1 or lookback - a >= MIN_AGE_DAYS:  # the extreme is at least four days old
            low[j] = lows[a]
        if lookback == 1 or lookback - b >= MIN_AGE_DAYS:
            high[j] = highs[b]
    return low, high


def signals(ctx: core.Context, params: dict) -> tuple[np.ndarray, np.ndarray]:
    day_low, day_high = levels(ctx.d1, params["lookback"])
    low, high = day_low[ctx.bar_day], day_high[ctx.bar_day]
    frame = pd.DataFrame({"day": ctx.fx_day, "h": ctx.h, "l": ctx.l}).groupby("day", sort=False)
    lowest, highest = frame["l"].cummin().to_numpy(), frame["h"].cummax().to_numpy()
    with np.errstate(invalid="ignore"):
        wanted = np.where((lowest < low) & (ctx.c > low), 1, np.where((highest > high) & (ctx.c < high), -1, 0))
    side = core.first_of_day(ctx.fx_day.astype(np.int64), wanted.astype(np.int8))
    return side, core.stops_by_side(side, lowest, highest)
