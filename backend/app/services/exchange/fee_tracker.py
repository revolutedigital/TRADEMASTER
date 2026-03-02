"""Fee tracking and analysis for exchange transactions."""
from decimal import Decimal
from dataclasses import dataclass

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.trade import Order

logger = get_logger(__name__)


@dataclass
class FeeReport:
    total_fees: float
    fees_by_symbol: dict[str, float]
    fee_as_pct_of_volume: float
    avg_fee_per_trade: float


class FeeTracker:
    """Track and analyze exchange fees across all trades."""

    async def get_total_fees(self, db: AsyncSession) -> Decimal:
        result = await db.execute(select(func.sum(Order.commission)))
        return Decimal(str(result.scalar() or 0))

    async def get_fees_by_symbol(self, db: AsyncSession) -> dict[str, float]:
        result = await db.execute(
            select(Order.symbol, func.sum(Order.commission))
            .group_by(Order.symbol)
        )
        return {row[0]: float(row[1] or 0) for row in result.all()}

    async def generate_report(self, db: AsyncSession) -> FeeReport:
        total = await self.get_total_fees(db)
        by_symbol = await self.get_fees_by_symbol(db)

        # Total volume
        vol_result = await db.execute(
            select(func.sum(Order.price * Order.quantity))
        )
        total_volume = float(vol_result.scalar() or 1)

        count_result = await db.execute(select(func.count(Order.id)))
        trade_count = count_result.scalar() or 1

        return FeeReport(
            total_fees=float(total),
            fees_by_symbol=by_symbol,
            fee_as_pct_of_volume=float(total) / total_volume * 100 if total_volume > 0 else 0,
            avg_fee_per_trade=float(total) / trade_count,
        )


fee_tracker = FeeTracker()
