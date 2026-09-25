"""Asian range breakout (the London-open breakout of the Tokyo session's range, a staple of FX day trading, e.g.
Kathy Lien, "Day Trading and Swing Trading the Currency Market", 2008): the range from 00:00 to 07:00 London; the
first close beyond it between 07:00 and 12:00 London trades, with the other side as the structural stop."""

from __future__ import annotations

import numpy as np

from scripts.research import fx_c10_core as core

PARAMS = [{"timeframe": "M5"}]


def signals(ctx: core.Context, params: dict) -> tuple[np.ndarray, np.ndarray]:
    minute, day = core.local_minutes(ctx.open_time, "Europe/London")
    high, low = core.window_extremes(ctx, minute < 7 * 60, day)
    active = (minute >= 7 * 60) & (minute < 12 * 60)
    side = core.first_break(day, active, ctx.c, high, low)
    return side, core.stops_by_side(side, low, high)
