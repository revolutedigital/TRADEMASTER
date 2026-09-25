"""Larry Williams' volatility breakout ("Long-Term Secrets to Short-Term Trading", 1999): buy when price closes
k times the previous day's range above the day's open, sell short k times below; the first break of the FX day
trades, with the day's open as the structural stop."""

from __future__ import annotations

import numpy as np

from scripts.research import fx_c10_core as core

PARAMS = [{"timeframe": "M5", "k": k} for k in (0.3, 0.5, 0.7)]


def signals(ctx: core.Context, params: dict) -> tuple[np.ndarray, np.ndarray]:
    reach = params["k"] * (ctx.prev_high - ctx.prev_low)
    active = np.ones(len(ctx.c), dtype=bool)
    side = core.first_break(ctx.fx_day, active, ctx.c, ctx.day_open + reach, ctx.day_open - reach)
    return side, core.stops_by_side(side, ctx.day_open, ctx.day_open)
