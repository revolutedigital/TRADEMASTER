"""Market intraday momentum (Gao, Han, Li and Zhou, "Market Intraday Momentum", Journal of Financial Economics 2018;
for currencies, Elaut, Frömmel and Lampaert, "Intraday momentum in FX markets", Journal of Financial Markets 2018).
The session's first half-hour return (previous session close to 30 minutes after the open) predicts the last
half-hour: at 30 minutes before the close, trade in its direction. `both` also requires the second-to-last half-hour
to agree (their r12 predictor). No structural stop."""

from __future__ import annotations

import numpy as np

from scripts.research import fx_c10_core as core

PARAMS = [{"timeframe": "M5", "rule": "first"}, {"timeframe": "M5", "rule": "both"}]


def signals(ctx: core.Context, params: dict) -> tuple[np.ndarray, np.ndarray]:
    first_half_hour = core.session_price(ctx, ctx.exchange_open + 30) - core.previous_session_close(ctx)
    before_last_hour = core.session_price(ctx, ctx.exchange_close - 60)
    at = ctx.exchange_minute == ctx.exchange_close - 35  # the bar that closes 30 minutes before the session close
    wanted = np.sign(first_half_hour)
    if params["rule"] == "both":
        wanted = np.where(np.sign(ctx.c - before_last_hour) == wanted, wanted, 0)
    side = np.where(at & np.isfinite(wanted), wanted, 0).astype(np.int8)
    return side, np.full(len(side), np.nan)
