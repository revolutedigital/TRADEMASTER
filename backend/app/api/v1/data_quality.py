"""Data quality monitoring API endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.dependencies import get_db
from app.services.data.quality import DataQualityMonitor

logger = get_logger(__name__)
router = APIRouter()


@router.get("/data-quality/report")
async def get_data_quality_report(db: AsyncSession = Depends(get_db)):
    """Get comprehensive data quality report."""
    monitor = DataQualityMonitor()
    report = await monitor.generate_report(db)
    return report


@router.get("/data-quality/freshness/{symbol}")
async def get_data_freshness(symbol: str, db: AsyncSession = Depends(get_db)):
    """Check data freshness for a specific symbol."""
    monitor = DataQualityMonitor()
    freshness = await monitor.check_freshness(db, symbol.upper())
    return {"symbol": symbol.upper(), "freshness": freshness}


@router.get("/data-quality/gaps/{symbol}")
async def get_data_gaps(symbol: str, interval: str = "1h", db: AsyncSession = Depends(get_db)):
    """Check for data gaps in a symbol's time series."""
    monitor = DataQualityMonitor()
    gaps = await monitor.check_gaps(db, symbol.upper(), interval)
    return {"symbol": symbol.upper(), "interval": interval, "gaps": gaps}
