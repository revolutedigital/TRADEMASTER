"""Tax reporting for trading activity."""
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import func, select, extract
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.trade import Order, Trade
from app.models.portfolio import Position

logger = get_logger(__name__)


@dataclass
class TaxSummary:
    year: int
    total_realized_gains: float
    total_realized_losses: float
    net_realized: float
    total_trades: int
    total_fees: float
    net_after_fees: float
    trades_by_month: dict[int, dict]


class TaxReporter:
    """Generate tax reports using FIFO method for cost basis."""

    async def _sum_commissions_for_position(
        self, db: AsyncSession, pos: Position
    ) -> float:
        """Sum Order + Trade commissions linked to a position by symbol and time window.

        Since Position has no direct FK to Order/Trade, we match orders by symbol
        whose created_at falls within [position.opened_at, position.closed_at].
        We also sum trade-level commissions through the Order->Trade FK.
        """
        if not pos.opened_at or not pos.closed_at:
            return 0.0

        # Sum Order-level commissions
        order_result = await db.execute(
            select(func.coalesce(func.sum(Order.commission), 0)).where(
                Order.symbol == pos.symbol,
                Order.created_at >= pos.opened_at,
                Order.created_at <= pos.closed_at,
            )
        )
        order_fees = float(order_result.scalar_one())

        # Sum Trade-level commissions (may differ from order-level, e.g. partial fills)
        trade_result = await db.execute(
            select(func.coalesce(func.sum(Trade.commission), 0)).where(
                Trade.symbol == pos.symbol,
                Trade.executed_at >= pos.opened_at,
                Trade.executed_at <= pos.closed_at,
            )
        )
        trade_fees = float(trade_result.scalar_one())

        # Use the higher of the two to avoid under-reporting fees.
        # Order.commission is the exchange-reported total; Trade.commission is per-fill.
        # They should match, but if they diverge we take the conservative (larger) value.
        return max(order_fees, trade_fees)

    async def generate_annual_report(self, db: AsyncSession, year: int) -> TaxSummary:
        """Generate tax summary for a given year."""
        # Get all closed positions for the year
        result = await db.execute(
            select(Position).where(
                Position.is_open == False,
                extract("year", Position.closed_at) == year,
            )
        )
        positions = result.scalars().all()

        total_gains = 0.0
        total_losses = 0.0
        total_fees = 0.0
        trades_by_month: dict[int, dict] = {}

        for pos in positions:
            pnl = float(pos.realized_pnl or 0)
            fees = await self._sum_commissions_for_position(db, pos)
            month = pos.closed_at.month if pos.closed_at else 0

            if pnl > 0:
                total_gains += pnl
            else:
                total_losses += abs(pnl)

            total_fees += fees

            if month not in trades_by_month:
                trades_by_month[month] = {"gains": 0, "losses": 0, "trades": 0, "fees": 0}

            trades_by_month[month]["trades"] += 1
            trades_by_month[month]["fees"] += fees
            if pnl > 0:
                trades_by_month[month]["gains"] += pnl
            else:
                trades_by_month[month]["losses"] += abs(pnl)

        net = total_gains - total_losses

        return TaxSummary(
            year=year,
            total_realized_gains=round(total_gains, 2),
            total_realized_losses=round(total_losses, 2),
            net_realized=round(net, 2),
            total_trades=len(positions),
            total_fees=round(total_fees, 2),
            net_after_fees=round(net - total_fees, 2),
            trades_by_month={k: {kk: round(vv, 2) for kk, vv in v.items()}
                           for k, v in sorted(trades_by_month.items())},
        )

    async def generate_csv_data(self, db: AsyncSession, year: int) -> list[dict]:
        """Generate CSV-compatible data for all trades in a year."""
        result = await db.execute(
            select(Position).where(
                Position.is_open == False,
                extract("year", Position.closed_at) == year,
            ).order_by(Position.closed_at)
        )
        positions = result.scalars().all()

        rows = []
        for pos in positions:
            commission = await self._sum_commissions_for_position(db, pos)
            realized = float(pos.realized_pnl or 0)
            rows.append({
                "date_opened": pos.created_at.isoformat() if pos.created_at else "",
                "date_closed": pos.closed_at.isoformat() if pos.closed_at else "",
                "symbol": pos.symbol,
                "side": pos.side,
                "quantity": float(pos.quantity or 0),
                "entry_price": float(pos.entry_price or 0),
                "exit_price": float(pos.current_price or 0),
                "realized_pnl": realized,
                "commission": round(commission, 8),
                "net_pnl": round(realized - commission, 8),
            })

        return rows


tax_reporter = TaxReporter()
