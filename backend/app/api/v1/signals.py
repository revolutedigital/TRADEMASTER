"""AI signals API endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db
from app.models.signal import PredictionSignal

router = APIRouter()


@router.get("/history")
async def get_signal_history(
    symbol: str | None = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """Get recent prediction signals."""
    query = (
        select(PredictionSignal)
        .order_by(PredictionSignal.generated_at.desc())
        .limit(limit)
    )
    if symbol:
        query = query.where(PredictionSignal.symbol == symbol.upper())
    result = await db.execute(query)
    signals = result.scalars().all()
    return [
        {
            "id": s.id,
            "symbol": s.symbol,
            "action": s.action,
            "strength": float(s.strength),
            "confidence": float(s.confidence),
            "model_source": s.model_source,
            "timeframe": s.timeframe,
            "was_executed": s.was_executed,
            "generated_at": s.generated_at.isoformat(),
        }
        for s in signals
    ]
