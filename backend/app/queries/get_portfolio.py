"""Query: Get portfolio summary."""

from app.core.logging import get_logger
from app.repositories.portfolio_repo import PortfolioRepository

logger = get_logger(__name__)


class GetPortfolioQuery:
    def __init__(self):
        self._repo = PortfolioRepository()

    async def execute(self, db) -> dict:
        snapshot = await self._repo.get_latest_snapshot(db)
        if not snapshot:
            return {"total_equity": 0, "positions": [], "daily_pnl": 0}
        return {
            "total_equity": float(snapshot.total_equity),
            "available_balance": float(snapshot.available_balance),
            "unrealized_pnl": float(snapshot.unrealized_pnl),
            "daily_pnl": float(snapshot.daily_pnl),
        }


get_portfolio_query = GetPortfolioQuery()
