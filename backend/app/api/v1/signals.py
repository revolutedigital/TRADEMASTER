"""AI signals API endpoints."""

import json

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, require_auth
from app.models.signal import PredictionSignal
from app.schemas.trading import SignalEvidenceResponse, SignalHistoryItemResponse

router = APIRouter()


@router.get("/history", response_model=list[SignalHistoryItemResponse])
async def get_signal_history(
    symbol: str | None = Query(default=None, pattern=r"^[A-Z0-9]{5,20}$"),
    limit: int = Query(default=50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
) -> list[SignalHistoryItemResponse]:
    """Get persisted strategy candidates with their explainable evidence."""
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
        SignalHistoryItemResponse(
            id=s.id,
            symbol=s.symbol,
            action=s.action,
            strength=float(s.strength),
            confidence=float(s.confidence),
            model_source=s.model_source,
            timeframe=s.timeframe,
            was_executed=s.was_executed,
            evidence=_parse_evidence(s.features_snapshot),
            generated_at=s.generated_at,
        )
        for s in signals
    ]


def _parse_evidence(raw_snapshot: str | None) -> SignalEvidenceResponse | None:
    """Return validated evidence while keeping legacy or malformed rows readable."""
    if not raw_snapshot:
        return None
    try:
        snapshot = json.loads(raw_snapshot)
        if not isinstance(snapshot, dict):
            return None
        return SignalEvidenceResponse.model_validate(snapshot)
    except (TypeError, ValueError):
        return None
