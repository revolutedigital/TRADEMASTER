"""Simplified single-asset intelligence endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, require_auth
from app.schemas.asset_intelligence import (
    AssetStudyCreateRequest,
    AssetStudyResponse,
    AssetUniverseResponse,
)
from app.services.asset_intelligence import AssetIntelligenceError, study_asset
from app.services.market.public_binance_data import PublicMarketDataUnavailable
from app.services.market.spot_asset_catalog import MIN_QUOTE_VOLUME_24H, spot_asset_catalog

router = APIRouter()


@router.get("/universe", response_model=AssetUniverseResponse)
async def list_eligible_assets(
    search: str = Query(default="", max_length=30),
    limit: int = Query(default=100, ge=1, le=500),
    _user: dict = Depends(require_auth),
) -> dict[str, object]:
    """Return the liquid Spot USDT universe for the asset picker."""
    try:
        assets, generated_at = await spot_asset_catalog.list(search=search, limit=limit)
    except PublicMarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "assets": [asset.as_dict() for asset in assets],
        "generated_at": generated_at.isoformat(),
        "minimum_quote_volume_24h": MIN_QUOTE_VOLUME_24H,
    }


@router.post("/studies", response_model=AssetStudyResponse, status_code=201)
async def create_asset_study(
    body: AssetStudyCreateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(require_auth),
) -> dict[str, object]:
    """Study, train, compare, and validate a strategy for one selected asset."""
    try:
        study = await study_asset(db, symbol=body.symbol)
        await db.commit()
    except ValueError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except AssetIntelligenceError as exc:
        await db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PublicMarketDataUnavailable as exc:
        await db.rollback()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception:
        await db.rollback()
        raise

    from app.core.audit import audit_logger

    await audit_logger.log_event(
        action="ASSET_INTELLIGENCE_STUDY_COMPLETED",
        user_id=str(user.get("sub", "admin")),
        resource=f"asset-intelligence:{study['symbol']}",
        details={
            "execution_mode": study["execution_mode"],
            "deployment_id": study["recommendation"]["deployment_id"],  # type: ignore[index]
            "deployment_status": study["recommendation"]["deployment_status"],  # type: ignore[index]
        },
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )
    return study
