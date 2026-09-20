"""What a forex bot did, in numbers a person can read: return, drawdown, time in position and where it wins.

Pure functions over closed trades, so the API, the reports and the tests share one definition. Money is in
the account currency; times are seconds since the epoch (UTC).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

LONDON = ZoneInfo("Europe/London")


@dataclass(frozen=True)
class ClosedTrade:
    symbol: str
    side: int
    entry_time: float
    exit_time: float
    pnl: float
    r_multiple: float | None = None
    exit_reason: str = ""


@dataclass(frozen=True)
class Slice:
    trades: int
    pnl: float


@dataclass(frozen=True)
class Performance:
    trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    return_pct: float
    profit_factor: float | None  # None when there are no losses to divide by
    avg_r: float | None
    max_drawdown_pct: float
    seconds_in_position: float
    share_of_time_in_position: float
    avg_hold_seconds: float
    by_hour_london: dict[int, Slice]
    by_weekday: dict[int, Slice]
    best: tuple[ClosedTrade, ...]
    worst: tuple[ClosedTrade, ...]


def _slices(trades: Sequence[ClosedTrade], key) -> dict[int, Slice]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for trade in trades:
        grouped[key(trade)].append(trade.pnl)
    return {k: Slice(len(v), sum(v)) for k, v in sorted(grouped.items())}


def max_drawdown_pct(pnls: Sequence[float], start_equity: float) -> float:
    """Largest peak-to-trough fall of the equity curve, in percent of the peak."""
    if start_equity <= 0:
        raise ValueError("start_equity must be positive")
    equity = peak = start_equity
    worst = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        worst = max(worst, (peak - equity) / peak)
    return 100.0 * worst


def _union_seconds(trades: Sequence[ClosedTrade]) -> float:
    """Seconds during which at least one position was open (overlaps counted once)."""
    total, end = 0.0, float("-inf")
    for trade in sorted(trades, key=lambda t: t.entry_time):
        start = max(trade.entry_time, end)
        if trade.exit_time > start:
            total += trade.exit_time - start
        end = max(end, trade.exit_time)
    return total


def performance(trades: Sequence[ClosedTrade], *, start_equity: float, top: int = 3) -> Performance:
    if start_equity <= 0:
        raise ValueError("start_equity must be positive")
    ordered = sorted(trades, key=lambda t: t.exit_time)
    wins = [t for t in ordered if t.pnl > 0]
    losses = [t for t in ordered if t.pnl < 0]
    gross_win, gross_loss = sum(t.pnl for t in wins), -sum(t.pnl for t in losses)
    total = sum(t.pnl for t in ordered)
    rs = [t.r_multiple for t in ordered if t.r_multiple is not None]
    in_position = _union_seconds(ordered)
    span = (max(t.exit_time for t in ordered) - min(t.entry_time for t in ordered)) if ordered else 0.0
    return Performance(
        trades=len(ordered),
        wins=len(wins),
        losses=len(losses),
        win_rate=len(wins) / len(ordered) if ordered else 0.0,
        total_pnl=total,
        return_pct=100.0 * total / start_equity,
        profit_factor=gross_win / gross_loss if gross_loss > 0 else None,
        avg_r=sum(rs) / len(rs) if rs else None,
        max_drawdown_pct=max_drawdown_pct([t.pnl for t in ordered], start_equity),
        seconds_in_position=in_position,
        share_of_time_in_position=in_position / span if span > 0 else 0.0,
        avg_hold_seconds=sum(t.exit_time - t.entry_time for t in ordered) / len(ordered) if ordered else 0.0,
        by_hour_london=_slices(ordered, lambda t: datetime.fromtimestamp(t.entry_time, UTC).astimezone(LONDON).hour),
        by_weekday=_slices(ordered, lambda t: datetime.fromtimestamp(t.entry_time, UTC).astimezone(LONDON).weekday()),
        best=tuple(sorted(ordered, key=lambda t: t.pnl, reverse=True)[:top]),
        worst=tuple(sorted(ordered, key=lambda t: t.pnl)[:top]),
    )
