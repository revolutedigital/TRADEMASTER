"""Market data collector: fetches historical OHLCV data and stores in database."""

from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.market import OHLCV
from app.services.exchange.binance_client import binance_client

logger = get_logger(__name__)


def _strip_tz(dt: datetime) -> datetime:
    """Convert timezone-aware datetime to naive UTC datetime for PostgreSQL."""
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


class MarketDataCollector:
    """Collects and stores historical and live market data."""

    async def seed_historical(
        self,
        db: AsyncSession,
        symbol: str,
        interval: str = "1h",
        days_back: int = 730,  # 2 years
    ) -> int:
        """Download historical kline data from Binance and store in database.

        Returns the number of candles inserted.
        """
        logger.info(
            "seeding_historical_data",
            symbol=symbol,
            interval=interval,
            days_back=days_back,
        )

        # Calculate start time
        start_ms = int(
            (datetime.now(timezone.utc).timestamp() - days_back * 86400) * 1000
        )
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        total_inserted = 0
        current_start = start_ms

        while current_start < end_ms:
            df = await binance_client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=1000,
                start_time=current_start,
            )

            if df.empty:
                break

            # Insert batch into database
            count = await self._insert_candles(db, df, symbol, interval)
            total_inserted += count

            # Move to next batch
            last_close_time = int(df["close_time"].iloc[-1].timestamp() * 1000)
            current_start = last_close_time + 1

            if len(df) < 1000:
                break

        await db.commit()
        logger.info(
            "historical_data_seeded",
            symbol=symbol,
            interval=interval,
            total_candles=total_inserted,
        )
        return total_inserted

    async def store_kline(
        self,
        db: AsyncSession,
        symbol: str,
        interval: str,
        data: dict,
    ) -> OHLCV | None:
        """Store a single kline (from WebSocket) into the database.

        Only stores closed candles.
        """
        if not data.get("is_closed"):
            return None

        candle = OHLCV(
            symbol=symbol,
            interval=interval,
            open_time=_strip_tz(pd.Timestamp(data["open_time"], unit="ms", tz="UTC").to_pydatetime()),
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            volume=data["volume"],
            close_time=_strip_tz(pd.Timestamp(data["close_time"], unit="ms", tz="UTC").to_pydatetime()),
            quote_volume=data.get("quote_volume", 0),
            trade_count=data.get("trade_count", 0),
        )
        db.add(candle)
        return candle

    async def _insert_candles(
        self,
        db: AsyncSession,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
    ) -> int:
        """Bulk insert candles from a DataFrame, skipping duplicates."""
        count = 0
        for _, row in df.iterrows():
            # Check for existing candle to avoid duplicates
            open_time = _strip_tz(row["open_time"].to_pydatetime())
            existing = await db.execute(
                select(OHLCV).where(
                    OHLCV.symbol == symbol,
                    OHLCV.interval == interval,
                    OHLCV.open_time == open_time,
                ).limit(1)
            )
            if existing.scalar_one_or_none():
                continue

            candle = OHLCV(
                symbol=symbol,
                interval=interval,
                open_time=open_time,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                close_time=_strip_tz(row["close_time"].to_pydatetime()),
                quote_volume=float(row["quote_volume"]),
                trade_count=int(row["trade_count"]),
            )
            db.add(candle)
            count += 1

        return count

    async def get_latest_candles(
        self,
        db: AsyncSession,
        symbol: str,
        interval: str = "1h",
        limit: int = 200,
    ) -> pd.DataFrame:
        """Fetch the latest candles from database as a DataFrame."""
        result = await db.execute(
            select(OHLCV)
            .where(OHLCV.symbol == symbol, OHLCV.interval == interval)
            .order_by(OHLCV.open_time.desc())
            .limit(limit)
        )
        candles = result.scalars().all()

        if not candles:
            return pd.DataFrame()

        data = [
            {
                "open_time": c.open_time,
                "open": float(c.open),
                "high": float(c.high),
                "low": float(c.low),
                "close": float(c.close),
                "volume": float(c.volume),
                "quote_volume": float(c.quote_volume),
                "trade_count": c.trade_count,
            }
            for c in reversed(candles)
        ]
        return pd.DataFrame(data)


market_data_collector = MarketDataCollector()
