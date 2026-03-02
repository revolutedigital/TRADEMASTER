"""Query: Get trade history."""

from app.core.logging import get_logger

logger = get_logger(__name__)


class GetTradeHistoryQuery:
    async def execute(self, db, symbol: str | None = None, limit: int = 50, offset: int = 0) -> list[dict]:
        from app.repositories.trade_repo import TradeRepository
        repo = TradeRepository()
        trades = await repo.get_recent(db, symbol=symbol, limit=limit, offset=offset)
        return [
            {
                "id": str(t.id),
                "symbol": t.symbol,
                "side": t.side,
                "quantity": float(t.quantity),
                "price": float(t.price),
                "status": t.status,
                "created_at": t.created_at.isoformat() if t.created_at else None,
            }
            for t in trades
        ]


get_trade_history_query = GetTradeHistoryQuery()
