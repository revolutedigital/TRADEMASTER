"""Intraday Donchian breakout with a daily trend filter (Richard Donchian's channel, the Turtle rules' entry on
intraday bars): buy a close above the highest high of the previous N bars when the previous daily close is above its
50-day average; sell short below the lowest low when it is below. Structural stop: the opposite N/2-bar channel (the
Turtles' exit channel)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.research import fx_c3_lab as c3
from scripts.research import fx_c10_core as core

PARAMS = [{"timeframe": tf, "n": n} for tf in ("M15", "H1") for n in (20, 55)]


def signals(ctx: core.Context, params: dict) -> tuple[np.ndarray, np.ndarray]:
    n = params["n"]
    upper, lower = c3.prior_max(ctx.h, n), c3.prior_min(ctx.l, n)
    with np.errstate(invalid="ignore"):
        side = np.where((ctx.c > upper) & (ctx.prev_close > ctx.sma50_d1), 1,
                        np.where((ctx.c < lower) & (ctx.prev_close < ctx.sma50_d1), -1, 0)).astype(np.int8)
    exit_low = pd.Series(ctx.l).rolling(n // 2).min().to_numpy()
    exit_high = pd.Series(ctx.h).rolling(n // 2).max().to_numpy()
    return side, core.stops_by_side(side, exit_low, exit_high)
