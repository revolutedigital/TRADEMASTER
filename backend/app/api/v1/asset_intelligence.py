"""Simplified single-asset intelligence endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, require_auth
from app.schemas.asset_intelligence import (
    AssetStudyCreateRequest,
    AssetStudyJobResponse,
    AssetUniverseResponse,
    MarketOpportunityScanResponse,
)
from app.services.asset_study_jobs import asset_study_job_service, serialize_study_job
from app.services.market_opportunity_scans import (
    market_opportunity_scan_service,
    serialize_scan,
)
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


@router.post("/studies", response_model=AssetStudyJobResponse, status_code=202)
async def create_asset_study(
    body: AssetStudyCreateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(require_auth),
) -> dict[str, object]:
    """Queue durable research so full training never exhausts the browser request."""
    started = False
    try:
        asset = await spot_asset_catalog.require(body.symbol)
        active_scan = await market_opportunity_scan_service.get_active(db)
        if active_scan is not None:
            raise HTTPException(
                status_code=409,
                detail=(
                    "A busca de oportunidades já está estudando candidatos. "
                    "Aguarde ela terminar antes de iniciar um estudo individual."
                ),
            )
        job, started = await asset_study_job_service.start_or_reuse(
            db,
            symbol=asset.symbol,
            requested_by=str(user.get("sub", "admin")),
        )
        if started:
            await db.commit()
    except ValueError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except PublicMarketDataUnavailable as exc:
        await db.rollback()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        await db.rollback()
        raise
    except Exception:
        await db.rollback()
        raise

    persisted_job = await asset_study_job_service.get(db, job.id)
    if persisted_job is None:
        raise HTTPException(status_code=404, detail="Asset study job was not found")

    if started:
        from app.core.audit import audit_logger

        await audit_logger.log_event(
            action="ASSET_INTELLIGENCE_STUDY_STARTED",
            user_id=str(user.get("sub", "admin")),
            resource=f"asset-study-job:{persisted_job.id}",
            details={"symbol": persisted_job.symbol, "execution_side_effects": "none"},
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
        asset_study_job_service.launch(persisted_job.id)

    return serialize_study_job(persisted_job)


@router.get("/studies/{study_id}", response_model=AssetStudyJobResponse)
async def get_asset_study(
    study_id: int = Path(ge=1),
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
) -> dict[str, object]:
    """Return progress and final evidence for a durable full-asset study."""
    job = await asset_study_job_service.get(db, study_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Asset study job was not found")
    return serialize_study_job(job)


@router.post("/opportunity-scans", response_model=MarketOpportunityScanResponse, status_code=202)
async def start_market_opportunity_scan(
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(require_auth),
) -> dict[str, object]:
    """Start one non-executing scan across the full eligible Spot universe."""
    started = False
    try:
        scan, started = await market_opportunity_scan_service.start_or_reuse(
            db,
            requested_by=str(user.get("sub", "admin")),
        )
        if started:
            await db.commit()
    except Exception:
        await db.rollback()
        raise

    persisted_scan = await market_opportunity_scan_service.get(db, scan.id)
    if persisted_scan is None:
        raise HTTPException(status_code=404, detail="Market opportunity scan was not found")

    if started:
        from app.core.audit import audit_logger

        await audit_logger.log_event(
            action="MARKET_OPPORTUNITY_SCAN_STARTED",
            user_id=str(user.get("sub", "admin")),
            resource=f"market-opportunity-scan:{persisted_scan.id}",
            details={"scope": "eligible_spot_usdt_catalog", "execution_side_effects": "none"},
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
        market_opportunity_scan_service.launch(persisted_scan.id)

    return serialize_scan(persisted_scan)


@router.get("/opportunity-scans/{scan_id}", response_model=MarketOpportunityScanResponse)
async def get_market_opportunity_scan(
    scan_id: int = Path(ge=1),
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
) -> dict[str, object]:
    """Return durable progress and research evidence for one market scan."""
    scan = await market_opportunity_scan_service.get(db, scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail="Market opportunity scan was not found")
    return serialize_scan(scan)
