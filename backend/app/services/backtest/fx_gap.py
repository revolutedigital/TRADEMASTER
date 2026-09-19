"""Stop and limit fills that respect weekend gaps and quote sides.

A backtest that fills every stop at the stop price is optimistic exactly when it
matters: when the market reopens on Sunday through the stop, or jumps on news, a real
stop order fills at the first available price, which is worse. Long positions exit on
the bid and short positions on the ask, so a stop is compared against that side.

This module has no exchange, database, or engine dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.services.backtest.fx_costs import Side


@dataclass(frozen=True)
class ProtectiveFill:
    """Result of testing a protective order against one bar."""

    triggered: bool
    price: float | None = None
    gapped: bool = False


NOT_TRIGGERED = ProtectiveFill(triggered=False)


def stop_fill(
    *,
    side: Side,
    stop_price: float,
    bar_open_bid: float,
    bar_open_ask: float,
    bar_low_bid: float,
    bar_high_ask: float,
) -> ProtectiveFill:
    """Fill a protective stop, filling at the gapped open when the bar opens through it.

    A long stop sells when the bid falls to the stop; a short stop buys when the ask
    rises to it.
    """
    if side == "LONG":
        if bar_open_bid <= stop_price:
            return ProtectiveFill(triggered=True, price=bar_open_bid, gapped=True)
        if bar_low_bid <= stop_price:
            return ProtectiveFill(triggered=True, price=stop_price)
        return NOT_TRIGGERED

    if bar_open_ask >= stop_price:
        return ProtectiveFill(triggered=True, price=bar_open_ask, gapped=True)
    if bar_high_ask >= stop_price:
        return ProtectiveFill(triggered=True, price=stop_price)
    return NOT_TRIGGERED


def take_profit_fill(
    *,
    side: Side,
    limit_price: float,
    bar_high_bid: float,
    bar_low_ask: float,
) -> ProtectiveFill:
    """Fill a take-profit at its limit price, never at a better gapped price.

    Assuming the limit price even when the market gaps beyond it keeps the backtest
    from crediting luck. A long limit sells into the bid; a short limit buys the ask.
    """
    if side == "LONG":
        if bar_high_bid >= limit_price:
            return ProtectiveFill(triggered=True, price=limit_price)
        return NOT_TRIGGERED
    if bar_low_ask <= limit_price:
        return ProtectiveFill(triggered=True, price=limit_price)
    return NOT_TRIGGERED


def resolve_protective_fill(stop: ProtectiveFill, take_profit: ProtectiveFill) -> ProtectiveFill:
    """When one bar touches both, assume the stop happened first (the pessimistic case)."""
    if stop.triggered:
        return stop
    return take_profit


def reopen_after_gap(index: pd.DatetimeIndex, *, min_gap_hours: float = 24.0) -> pd.Series:
    """Flag the first bar after a market closure, such as the Sunday reopen."""
    if not index.is_monotonic_increasing:
        raise ValueError("index must be sorted ascending")
    gaps = index.to_series().diff()
    return (gaps >= pd.Timedelta(hours=min_gap_hours)).rename("reopen_after_gap")
