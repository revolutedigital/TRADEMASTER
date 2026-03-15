"""Data quality monitoring for market data."""
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.market import OHLCV

logger = get_logger(__name__)


@dataclass
class DataQualityReport:
    symbol: str
    interval: str
    total_candles: int
    freshness_seconds: float | None
    gap_count: int
    gaps: list[dict]  # [{start, end, missing_candles}]
    duplicate_count: int
    anomaly_count: int  # price anomalies (e.g., close > 2x open)
    quality_score: float  # 0-100


class DataQualityMonitor:
    """Monitor data quality for market data."""

    # Expected interval durations in seconds
    INTERVAL_SECONDS = {
        "1m": 60, "3m": 180, "5m": 300, "15m": 900,
        "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400,
        "6h": 21600, "8h": 28800, "12h": 43200, "1d": 86400,
    }

    async def check_freshness(self, db: AsyncSession, symbol: str, interval: str = "1h") -> float | None:
        """Check how many seconds since the last candle."""
        result = await db.execute(
            select(func.max(OHLCV.open_time)).where(
                OHLCV.symbol == symbol,
                OHLCV.interval == interval,
            )
        )
        latest = result.scalar()
        if latest is None:
            return None

        now = datetime.now(timezone.utc)
        if latest.tzinfo is None:
            from datetime import timezone as tz
            latest = latest.replace(tzinfo=tz.utc)
        return (now - latest).total_seconds()

    async def check_gaps(self, db: AsyncSession, symbol: str, interval: str = "1h",
                         days_back: int = 30) -> list[dict]:
        """Find temporal gaps in candle data."""
        interval_sec = self.INTERVAL_SECONDS.get(interval, 3600)
        tolerance = interval_sec * 1.5  # Allow 50% tolerance

        since = datetime.now(timezone.utc) - timedelta(days=days_back)
        result = await db.execute(
            select(OHLCV.open_time).where(
                OHLCV.symbol == symbol,
                OHLCV.interval == interval,
                OHLCV.open_time >= since,
            ).order_by(OHLCV.open_time)
        )
        times = [row[0] for row in result.all()]

        gaps = []
        for i in range(1, len(times)):
            prev = times[i - 1]
            curr = times[i]
            # Ensure both are timezone-aware for comparison
            if prev.tzinfo is None:
                prev = prev.replace(tzinfo=timezone.utc)
            if curr.tzinfo is None:
                curr = curr.replace(tzinfo=timezone.utc)

            diff = (curr - prev).total_seconds()
            if diff > tolerance:
                missing = int(diff / interval_sec) - 1
                gaps.append({
                    "start": prev.isoformat(),
                    "end": curr.isoformat(),
                    "gap_seconds": diff,
                    "missing_candles": missing,
                })

        return gaps

    async def check_duplicates(self, db: AsyncSession, symbol: str, interval: str = "1h") -> int:
        """Count duplicate candles (same open_time)."""
        result = await db.execute(
            text("""
                SELECT COUNT(*) FROM (
                    SELECT open_time, COUNT(*) as cnt
                    FROM ohlcv
                    WHERE symbol = :symbol AND interval = :interval
                    GROUP BY open_time
                    HAVING COUNT(*) > 1
                ) dupes
            """),
            {"symbol": symbol, "interval": interval},
        )
        return result.scalar() or 0

    async def check_anomalies(self, db: AsyncSession, symbol: str, interval: str = "1h",
                               days_back: int = 30) -> int:
        """Count price anomalies (close > 2x open, volume = 0, etc)."""
        since = datetime.now(timezone.utc) - timedelta(days=days_back)
        result = await db.execute(
            select(func.count(OHLCV.id)).where(
                OHLCV.symbol == symbol,
                OHLCV.interval == interval,
                OHLCV.open_time >= since,
            ).where(
                # Anomaly: close is more than 50% different from open
                (OHLCV.close > OHLCV.open * 1.5) | (OHLCV.close < OHLCV.open * 0.5) |
                # Or volume is zero (suspicious)
                (OHLCV.volume == 0)
            )
        )
        return result.scalar() or 0

    async def generate_report(self, db: AsyncSession, symbol: str, interval: str = "1h",
                               days_back: int = 30) -> DataQualityReport:
        """Generate comprehensive data quality report."""
        # Total candles
        since = datetime.now(timezone.utc) - timedelta(days=days_back)
        count_result = await db.execute(
            select(func.count(OHLCV.id)).where(
                OHLCV.symbol == symbol,
                OHLCV.interval == interval,
                OHLCV.open_time >= since,
            )
        )
        total = count_result.scalar() or 0

        freshness = await self.check_freshness(db, symbol, interval)
        gaps = await self.check_gaps(db, symbol, interval, days_back)
        duplicates = await self.check_duplicates(db, symbol, interval)
        anomalies = await self.check_anomalies(db, symbol, interval, days_back)

        # Calculate quality score (0-100)
        score = 100.0

        # Freshness penalty: -20 if stale (> 2x interval)
        interval_sec = self.INTERVAL_SECONDS.get(interval, 3600)
        if freshness is not None and freshness > interval_sec * 2:
            score -= min(20, (freshness / interval_sec - 2) * 5)

        # Gap penalty: -2 per gap, max -30
        gap_count = len(gaps)
        score -= min(30, gap_count * 2)

        # Duplicate penalty: -1 per duplicate, max -10
        score -= min(10, duplicates)

        # Anomaly penalty: -2 per anomaly, max -20
        score -= min(20, anomalies * 2)

        # Coverage penalty: expected vs actual candles
        expected = days_back * 24 * 3600 / interval_sec
        if expected > 0 and total < expected * 0.8:
            coverage_ratio = total / expected
            score -= (1 - coverage_ratio) * 20

        return DataQualityReport(
            symbol=symbol,
            interval=interval,
            total_candles=total,
            freshness_seconds=round(freshness, 1) if freshness else None,
            gap_count=gap_count,
            gaps=gaps[:20],  # Limit to 20 gaps
            duplicate_count=duplicates,
            anomaly_count=anomalies,
            quality_score=round(max(0, score), 1),
        )


data_quality_monitor = DataQualityMonitor()
