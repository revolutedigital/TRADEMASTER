"""Data quality monitoring endpoints."""
from fastapi import APIRouter, Depends, Query

from app.config import settings
from app.dependencies import require_auth
from app.models.base import async_session_factory

router = APIRouter()


@router.get("/data-quality/{symbol}")
async def get_data_quality(
    symbol: str,
    interval: str = Query(default="1h", pattern="^(1m|5m|15m|30m|1h|4h|1d)$"),
    days_back: int = Query(default=30, ge=1, le=365),
    _user: dict = Depends(require_auth),
):
    """Get data quality report for a symbol."""
    from app.services.data.quality import data_quality_monitor
    async with async_session_factory() as db:
        report = await data_quality_monitor.generate_report(
            db, symbol.upper(), interval, days_back,
        )
        return {
            "symbol": report.symbol,
            "interval": report.interval,
            "total_candles": report.total_candles,
            "freshness_seconds": report.freshness_seconds,
            "gap_count": report.gap_count,
            "gaps": report.gaps,
            "duplicate_count": report.duplicate_count,
            "anomaly_count": report.anomaly_count,
            "quality_score": report.quality_score,
        }


@router.get("/data-quality")
async def get_all_data_quality(
    interval: str = Query(default="1h", pattern="^(1m|5m|15m|30m|1h|4h|1d)$"),
    _user: dict = Depends(require_auth),
):
    """Get data quality summary for all configured symbols."""
    from app.services.data.quality import data_quality_monitor

    results = []
    async with async_session_factory() as db:
        for symbol in settings.symbols_list:
            report = await data_quality_monitor.generate_report(db, symbol, interval, 7)
            results.append({
                "symbol": report.symbol,
                "quality_score": report.quality_score,
                "total_candles": report.total_candles,
                "freshness_seconds": report.freshness_seconds,
                "gap_count": report.gap_count,
                "duplicate_count": report.duplicate_count,
            })

    avg_score = sum(r["quality_score"] for r in results) / len(results) if results else 0
    return {
        "overall_score": round(avg_score, 1),
        "symbols": results,
    }
