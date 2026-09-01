"""Authenticated, evidence-gated technical strategy deployment endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import TradingExecutionMode, settings
from app.dependencies import get_db, get_trading_engine, require_auth
from app.models.strategy_deployment import StrategyDeployment
from app.schemas.trading import (
    StrategyDeploymentActivationRequest,
    StrategyDeploymentCreateRequest,
    StrategyDeploymentResponse,
)
from app.services.exchange.live_trading_guard import (
    LiveTradingSafetyError,
    live_trading_guard,
)
from app.services.strategy_deployments import (
    StrategyDeploymentSourceError,
    activate_strategy_deployment,
    create_strategy_deployment,
)

router = APIRouter()


@router.get("", response_model=list[StrategyDeploymentResponse])
async def list_strategy_deployments(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),  # noqa: B008
    _user: dict = Depends(require_auth),  # noqa: B008
) -> list[StrategyDeployment]:
    """List stored approval evidence, newest first."""
    result = await db.execute(
        select(StrategyDeployment)
        .order_by(StrategyDeployment.created_at.desc(), StrategyDeployment.id.desc())
        .limit(min(max(limit, 1), 100))
    )
    return list(result.scalars().all())


@router.post("", response_model=StrategyDeploymentResponse, status_code=201)
async def create_deployment(
    body: StrategyDeploymentCreateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),  # noqa: B008
    user: dict = Depends(require_auth),  # noqa: B008
) -> StrategyDeployment:
    """Store fresh out-of-sample evidence; it never enables trading by itself."""
    try:
        deployment = await create_strategy_deployment(
            db,
            source_backtest_id=body.source_backtest_id,
            target_execution_mode=body.target_execution_mode,
        )
        await db.commit()
        await db.refresh(deployment)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except StrategyDeploymentSourceError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception:
        await db.rollback()
        raise

    from app.core.audit import audit_logger

    await audit_logger.log_event(
        action="STRATEGY_DEPLOYMENT_VALIDATED",
        user_id=str(user.get("sub", "admin")),
        resource=f"strategy-deployment:{deployment.id}",
        details={
            "source_backtest_id": deployment.source_backtest_id,
            "target_execution_mode": deployment.target_execution_mode,
            "status": deployment.status,
            "rejection_reason": deployment.rejection_reason,
        },
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )
    return deployment


@router.post("/{deployment_id}/activate", response_model=StrategyDeploymentResponse)
async def activate_deployment(
    deployment_id: int,
    body: StrategyDeploymentActivationRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),  # noqa: B008
    user: dict = Depends(require_auth),  # noqa: B008
    engine=Depends(get_trading_engine),
) -> StrategyDeployment:
    """Make an approved strategy available to the matching engine mode."""
    if engine._running:
        raise HTTPException(
            status_code=409,
            detail="stop the trading engine before activating a strategy deployment",
        )
    if settings.execution_mode == TradingExecutionMode.LIVE:
        try:
            live_trading_guard.require_live_exit(body.totp_code or "")
        except LiveTradingSafetyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    try:
        deployment = await activate_strategy_deployment(
            db,
            deployment_id=deployment_id,
            execution_mode=settings.execution_mode,
        )
        if settings.execution_mode != TradingExecutionMode.PAPER:
            from app.services.exchange.binance_ws import binance_ws_manager

            try:
                await binance_ws_manager.ensure_symbol(deployment.symbol)
            except RuntimeError as exc:
                raise HTTPException(
                    status_code=409,
                    detail="Binance market data is not active for this asset; strategy remains inactive",
                ) from exc
        await db.commit()
        await db.refresh(deployment)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except StrategyDeploymentSourceError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception:
        await db.rollback()
        raise

    from app.core.audit import audit_logger

    await audit_logger.log_event(
        action="STRATEGY_DEPLOYMENT_ACTIVATED",
        user_id=str(user.get("sub", "admin")),
        resource=f"strategy-deployment:{deployment.id}",
        details={
            "source_backtest_id": deployment.source_backtest_id,
            "target_execution_mode": deployment.target_execution_mode,
            "symbol": deployment.symbol,
            "interval": deployment.interval,
        },
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )
    return deployment
