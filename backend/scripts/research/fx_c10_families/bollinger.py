"""Bollinger band fade (John Bollinger, "Bollinger on Bollinger Bands", 2001, the mean-reversion use of the bands):
buy a close below the lower band (20-bar average minus k standard deviations), sell a close above the upper band.
No structural stop."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.research import fx_c10_core as core

PARAMS = [{"timeframe": tf, "width": width} for tf in ("M15", "M30") for width in (2.0, 2.5)]


def signals(ctx: core.Context, params: dict) -> tuple[np.ndarray, np.ndarray]:
    close = pd.Series(ctx.c)
    mid, deviation = close.rolling(20).mean().to_numpy(), close.rolling(20).std(ddof=0).to_numpy()
    with np.errstate(invalid="ignore"):
        side = np.where(ctx.c < mid - params["width"] * deviation, 1,
                        np.where(ctx.c > mid + params["width"] * deviation, -1, 0)).astype(np.int8)
    return side, np.full(len(side), np.nan)
