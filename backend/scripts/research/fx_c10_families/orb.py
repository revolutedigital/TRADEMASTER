"""Opening range breakout (Toby Crabel, "Day Trading with Short Term Price Patterns and Opening Range Breakout", 1990).
The range is the first N minutes of the market's session (the exchange's cash open for an index, London's 08:00 for a
currency pair); the first bar that closes beyond it within four hours of the open trades in that direction, with the
other side of the range as the structural stop. One signal a day. The `trend` variant only takes the side of the
daily trend (previous close against its 50-day average).

`first_candle` is the 5-minute version of Zarattini, Barbon and Aziz ("A Profitable Day Trading Strategy for the U.S.
Equity Market", 2024): the direction of the session's first 5-minute candle, stop at its other extreme."""

from __future__ import annotations

import numpy as np

from scripts.research import fx_c10_core as core

PARAMS = [{"timeframe": "M5", "range_minutes": 15}, {"timeframe": "M5", "range_minutes": 30},
          {"timeframe": "M5", "range_minutes": 60}, {"timeframe": "M5", "range_minutes": 30, "trend": True},
          {"timeframe": "M5", "first_candle": True}]
WINDOW_MINUTES = 240


def signals(ctx: core.Context, params: dict) -> tuple[np.ndarray, np.ndarray]:
    if params.get("first_candle"):
        first = ctx.exchange_minute == ctx.exchange_open
        side = np.where(first, np.sign(ctx.c - ctx.o), 0).astype(np.int8)
        return side, core.stops_by_side(side, ctx.l, ctx.h)
    start = ctx.exchange_open
    end = start + params["range_minutes"]
    high, low = core.window_extremes(ctx, core.in_session(ctx, start, end), ctx.exchange_day)
    active = core.in_session(ctx, end, start + WINDOW_MINUTES)  # only bars that open after the whole range
    up, down = high, low
    if params.get("trend"):
        with np.errstate(invalid="ignore"):
            up = np.where(ctx.prev_close > ctx.sma50_d1, high, np.inf)
            down = np.where(ctx.prev_close < ctx.sma50_d1, low, -np.inf)
    side = core.first_break(ctx.exchange_day, active, ctx.c, up, down)
    return side, core.stops_by_side(side, low, high)
