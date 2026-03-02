"""Time-travel queries: query data as it existed at a specific point in time.

Implements a simplified version of Delta Lake / Apache Iceberg concepts
using PostgreSQL temporal tables pattern.
"""

from datetime import datetime, timezone
from sqlalchemy import text

from app.core.logging import get_logger

logger = get_logger(__name__)


class TimeTravelQuery:
    """Execute queries against historical data snapshots."""

    async def get_portfolio_at(self, db, timestamp: datetime) -> dict:
        """Get portfolio state as it was at a specific timestamp."""
        result = await db.execute(
            text("""
                SELECT total_equity, available_balance, unrealized_pnl, daily_pnl
                FROM portfolio_snapshots
                WHERE created_at <= :ts
                ORDER BY created_at DESC
                LIMIT 1
            """),
            {"ts": timestamp},
        )
        row = result.first()
        if not row:
            return {"total_equity": 0, "available_balance": 0, "unrealized_pnl": 0, "daily_pnl": 0}
        return {
            "total_equity": float(row[0]),
            "available_balance": float(row[1]),
            "unrealized_pnl": float(row[2]),
            "daily_pnl": float(row[3]),
            "as_of": timestamp.isoformat(),
        }

    async def get_positions_at(self, db, timestamp: datetime) -> list[dict]:
        """Get open positions as they were at a specific timestamp."""
        result = await db.execute(
            text("""
                SELECT symbol, side, quantity, entry_price, created_at
                FROM positions
                WHERE created_at <= :ts AND (closed_at IS NULL OR closed_at > :ts)
                ORDER BY created_at DESC
            """),
            {"ts": timestamp},
        )
        return [
            {
                "symbol": row[0],
                "side": row[1],
                "quantity": float(row[2]),
                "entry_price": float(row[3]),
                "opened_at": row[4].isoformat() if row[4] else None,
            }
            for row in result.fetchall()
        ]

    async def get_market_data_at(self, db, symbol: str, timestamp: datetime, interval: str = "1h") -> dict | None:
        """Get market data candle at a specific timestamp."""
        result = await db.execute(
            text("""
                SELECT open, high, low, close, volume, open_time
                FROM ohlcv
                WHERE symbol = :symbol AND interval = :interval AND open_time <= :ts
                ORDER BY open_time DESC
                LIMIT 1
            """),
            {"symbol": symbol, "interval": interval, "ts": timestamp},
        )
        row = result.first()
        if not row:
            return None
        return {
            "open": float(row[0]),
            "high": float(row[1]),
            "low": float(row[2]),
            "close": float(row[3]),
            "volume": float(row[4]),
            "open_time": row[5].isoformat() if row[5] else None,
        }

    async def compare_snapshots(self, db, timestamp1: datetime, timestamp2: datetime) -> dict:
        """Compare portfolio state between two points in time."""
        snapshot1 = await self.get_portfolio_at(db, timestamp1)
        snapshot2 = await self.get_portfolio_at(db, timestamp2)

        equity_change = snapshot2["total_equity"] - snapshot1["total_equity"]
        equity_pct = (equity_change / snapshot1["total_equity"] * 100) if snapshot1["total_equity"] > 0 else 0

        return {
            "from": timestamp1.isoformat(),
            "to": timestamp2.isoformat(),
            "snapshot_from": snapshot1,
            "snapshot_to": snapshot2,
            "equity_change": equity_change,
            "equity_change_pct": round(equity_pct, 2),
        }


time_travel = TimeTravelQuery()
