"""Previous-day high/low breakout (the classic continuation break of yesterday's extremes): during the session, the
first close above the previous FX day's high buys, below its low sells short; the day's opposite extreme so far is
the structural stop."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.research import fx_c10_core as core

PARAMS = [{"timeframe": "M5"}]


def signals(ctx: core.Context, params: dict) -> tuple[np.ndarray, np.ndarray]:
    session = core.in_session(ctx, ctx.exchange_open, ctx.exchange_close)
    side = core.first_break(ctx.fx_day, session, ctx.c, ctx.prev_high, ctx.prev_low)
    frame = pd.DataFrame({"day": ctx.fx_day, "h": ctx.h, "l": ctx.l}).groupby("day", sort=False)
    return side, core.stops_by_side(side, frame["l"].cummin().to_numpy(), frame["h"].cummax().to_numpy())
