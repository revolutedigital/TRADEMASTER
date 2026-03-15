"""Live performance tracker: computes win rate, avg win/loss from closed positions.

Used by Kelly criterion position sizing to adapt position sizes based on
actual historical performance rather than fixed fractions.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.portfolio import Position

logger = get_logger(__name__)

MIN_TRADES_FOR_KELLY = 20  # Need at least 20 closed trades for reliable stats


@dataclass
class PerformanceStats:
    """Live trading performance statistics."""

    total_closed: int
    wins: int
    losses: int
    win_rate: float
    avg_win: float  # Average winning P&L (absolute)
    avg_loss: float  # Average losing P&L (absolute, positive number)
    expectancy: float  # Expected value per trade
    profit_factor: float  # Gross profit / gross loss
    has_enough_data: bool  # >= MIN_TRADES_FOR_KELLY


class PerformanceTracker:
    """Computes live performance stats from closed positions in DB."""

    async def get_stats(
        self,
        db: AsyncSession,
        symbol: str | None = None,
        lookback_days: int = 90,
    ) -> PerformanceStats:
        """Calculate performance stats from closed positions.

        Args:
            db: Database session
            symbol: Filter by symbol (None = all symbols)
            lookback_days: Only consider trades from the last N days
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        filters = [
            Position.is_open == False,
            Position.closed_at >= cutoff,
            Position.realized_pnl != None,
        ]
        if symbol:
            filters.append(Position.symbol == symbol)

        result = await db.execute(
            select(Position).where(and_(*filters)).order_by(Position.closed_at.desc())
        )
        positions = list(result.scalars().all())

        total_closed = len(positions)
        if total_closed == 0:
            return PerformanceStats(
                total_closed=0, wins=0, losses=0,
                win_rate=0, avg_win=0, avg_loss=0,
                expectancy=0, profit_factor=0,
                has_enough_data=False,
            )

        wins = [p for p in positions if float(p.realized_pnl) > 0]
        losses = [p for p in positions if float(p.realized_pnl) <= 0]

        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total_closed

        avg_win = (
            sum(float(p.realized_pnl) for p in wins) / win_count
            if win_count > 0
            else 0
        )
        avg_loss = (
            abs(sum(float(p.realized_pnl) for p in losses) / loss_count)
            if loss_count > 0
            else 0
        )

        gross_profit = sum(float(p.realized_pnl) for p in wins)
        gross_loss = abs(sum(float(p.realized_pnl) for p in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        stats = PerformanceStats(
            total_closed=total_closed,
            wins=win_count,
            losses=loss_count,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            expectancy=expectancy,
            profit_factor=profit_factor,
            has_enough_data=total_closed >= MIN_TRADES_FOR_KELLY,
        )

        logger.info(
            "performance_stats_computed",
            total=total_closed,
            win_rate=round(win_rate, 4),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            expectancy=round(expectancy, 2),
            profit_factor=round(profit_factor, 2),
            kelly_eligible=stats.has_enough_data,
        )

        return stats


performance_tracker = PerformanceTracker()
