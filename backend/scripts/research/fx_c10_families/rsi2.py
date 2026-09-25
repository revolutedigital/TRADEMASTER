"""Connors RSI(2) pullback (Larry Connors and Cesar Alvarez, "Short Term Trading Strategies That Work", 2008), on
intraday bars: buy when RSI(2) closes below the low threshold with the close above its 200-bar average, sell short
when it closes above the high threshold below the average. No structural stop."""

from __future__ import annotations

import numpy as np

from scripts.research import fx_c3_lab as c3
from scripts.research import fx_c10_core as core

PARAMS = [{"timeframe": tf, "low": low, "high": 100 - low} for tf in ("M15", "M30") for low in (5, 10)]


def signals(ctx: core.Context, params: dict) -> tuple[np.ndarray, np.ndarray]:
    rsi, trend = c3.rsi(ctx.c, 2), c3.sma(ctx.c, 200)
    with np.errstate(invalid="ignore"):
        side = np.where((rsi < params["low"]) & (ctx.c > trend), 1,
                        np.where((rsi > params["high"]) & (ctx.c < trend), -1, 0)).astype(np.int8)
    return side, np.full(len(side), np.nan)
