"""Narrow-range breakout (Toby Crabel, 1990): after a day whose range is the smallest of the last N days (NR4, NR7),
trade the first close beyond that day's high or low; the other extreme of that day is the structural stop."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.research import fx_c10_core as core

PARAMS = [{"timeframe": "M5", "n": 4}, {"timeframe": "M5", "n": 7}]


def signals(ctx: core.Context, params: dict) -> tuple[np.ndarray, np.ndarray]:
    day_range = pd.Series(ctx.d1[:, 1] - ctx.d1[:, 2])
    narrow = (day_range <= day_range.rolling(params["n"]).min()).to_numpy()  # each day against itself and the N-1 before
    previous = ctx.bar_day - 1
    active = np.zeros(len(ctx.c), dtype=bool)
    ok = previous >= 0
    active[ok] = narrow[previous[ok]]
    side = core.first_break(ctx.fx_day, active, ctx.c, ctx.prev_high, ctx.prev_low)
    return side, core.stops_by_side(side, ctx.prev_low, ctx.prev_high)
