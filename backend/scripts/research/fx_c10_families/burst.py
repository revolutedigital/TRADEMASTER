"""Momentum burst and spike fade: a bar whose range is at least m times the average range of the previous 20 bars and
that closes in its outer quarter, during the market's session. `follow` trades in its direction with the bar's other
extreme as the structural stop (Pradeep Bonde's momentum burst); `fade` trades against it with a stop beyond the
bar's extreme (the exhaustion-spike reversal)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.research import fx_c10_core as core

PARAMS = [{"timeframe": "M5", "size": size, "mode": mode} for mode in ("follow", "fade") for size in (2.5, 4.0)]


def signals(ctx: core.Context, params: dict) -> tuple[np.ndarray, np.ndarray]:
    bar_range = ctx.h - ctx.l
    average = pd.Series(bar_range).rolling(20).mean().shift(1).to_numpy()
    session = core.in_session(ctx, ctx.exchange_open, ctx.exchange_close)
    with np.errstate(invalid="ignore", divide="ignore"):
        big = session & (bar_range >= params["size"] * average) & (bar_range > 0)
        place = (ctx.c - ctx.l) / bar_range
    direction = np.where(big & (place >= 0.75), 1, np.where(big & (place <= 0.25), -1, 0))
    side = (direction if params["mode"] == "follow" else -direction).astype(np.int8)
    # either way a long's stop is under the bar's low and a short's over its high
    return side, core.stops_by_side(side, ctx.l, ctx.h)
