"""Market data repository: data access for OHLCV candles."""

from datetime import datetime

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.market import OHLCV
from app.repositories.base import BaseRepository


class MarketDataRepository(BaseRepository[OHLCV]):
    def __init__(self) -> None:
        super().__init__(OHLCV)

    async def get_latest_candles(
        self,
        db: AsyncSession,
        symbol: str,
        interval: str,
        limit: int = 300,
    ) -> pd.DataFrame:
        result = await db.execute(
            select(OHLCV)
            .where(OHLCV.symbol == symbol, OHLCV.interval == interval)
            .order_by(OHLCV.open_time.desc())
            .limit(limit)
        )
        rows = result.scalars().all()

        if not rows:
            return pd.DataFrame()

        data = [
            {
                "open_time": r.open_time,
                "open": float(r.open),
                "high": float(r.high),
                "low": float(r.low),
                "close": float(r.close),
                "volume": float(r.volume),
            }
            for r in reversed(rows)
        ]
        return pd.DataFrame(data)

    async def get_latest_price(
        self, db: AsyncSession, symbol: str
    ) -> float | None:
        result = await db.execute(
            select(OHLCV)
            .where(OHLCV.symbol == symbol, OHLCV.interval == "1m")
            .order_by(OHLCV.open_time.desc())
            .limit(1)
        )
        candle = result.scalar_one_or_none()
        return float(candle.close) if candle else None

    async def get_candles_since(
        self,
        db: AsyncSession,
        symbol: str,
        interval: str,
        since: datetime,
    ) -> list[OHLCV]:
        result = await db.execute(
            select(OHLCV)
            .where(
                OHLCV.symbol == symbol,
                OHLCV.interval == interval,
                OHLCV.open_time >= since,
            )
            .order_by(OHLCV.open_time.asc())
        )
        return list(result.scalars().all())
