"""Opening gap fade (the overnight-reversal literature: e.g. Plastun, Sibande, Gupta and Wohar, "Price gap
anomaly in the US stock market", 2020): when the session opens more than g daily ATRs away from the previous
session's close, trade back toward that close at the close of the session's first 5-minute bar. Currency pairs use
London's session. No structural stop."""

from __future__ import annotations

import numpy as np

from scripts.research import fx_c10_core as core

PARAMS = [{"timeframe": "M5", "gap_atr": g} for g in (0.25, 0.5)]


def signals(ctx: core.Context, params: dict) -> tuple[np.ndarray, np.ndarray]:
    first = ctx.exchange_minute == ctx.exchange_open
    gap = ctx.o - core.previous_session_close(ctx)
    with np.errstate(invalid="ignore"):
        side = np.where(first & (np.abs(gap) > params["gap_atr"] * ctx.atr_d1), -np.sign(gap), 0).astype(np.int8)
    return side, np.full(len(side), np.nan)
