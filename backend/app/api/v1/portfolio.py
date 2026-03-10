"""Portfolio API endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import (
    get_db,
    get_position_repository,
    require_auth,
    get_binance_client,
)
from app.repositories.position_repo import PositionRepository
from app.schemas.trading import PositionResponse, PortfolioSummary
from app.services.risk.drawdown import circuit_breaker
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/positions", response_model=list[PositionResponse])
async def get_positions(
    symbol: str | None = None,
    is_open: bool = True,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
    repo: PositionRepository = Depends(get_position_repository),
):
    """Get positions."""
    if is_open:
        positions = await repo.get_open(db, symbol)
    else:
        positions = await repo.get_closed(db, symbol, limit)
    return positions[:limit]


@router.get("/summary", response_model=PortfolioSummary)
async def get_portfolio_summary(
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
    repo: PositionRepository = Depends(get_position_repository),
):
    """Get portfolio summary with all open positions."""
    open_positions = await repo.get_open(db)
    closed_positions = await repo.get_closed(db, limit=1000)

    total_unrealized = sum(float(p.unrealized_pnl) for p in open_positions)
    total_exposure = sum(float(p.current_price) * float(p.quantity) for p in open_positions)
    total_realized = sum(float(p.realized_pnl) for p in closed_positions)

    # Query live balance from Binance
    total_equity = 0.0
    available_balance = 0.0
    try:
        exchange = get_binance_client()
        balance = await exchange.get_balance("USDT")
        available_balance = float(balance)
        total_equity = available_balance + total_exposure
    except Exception as e:
        logger.warning("balance_fetch_failed", error=str(e))
        total_equity = total_exposure

    daily_pnl = total_unrealized + total_realized
    exposure_pct = total_exposure / total_equity if total_equity > 0 else 0.0
    daily_pnl_pct = daily_pnl / total_equity if total_equity > 0 else 0.0

    return PortfolioSummary(
        total_equity=total_equity,
        available_balance=available_balance,
        total_unrealized_pnl=total_unrealized,
        total_realized_pnl=total_realized,
        total_exposure=total_exposure,
        exposure_pct=exposure_pct,
        open_positions=len(open_positions),
        daily_pnl=daily_pnl,
        daily_pnl_pct=daily_pnl_pct,
    )


@router.get("/risk-status")
async def get_risk_status(_user: dict = Depends(require_auth)):
    """Get current risk management status."""
    return circuit_breaker.get_status()


@router.get("/risk-metrics")
async def get_risk_metrics(
    _user: dict = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
    repo: PositionRepository = Depends(get_position_repository),
):
    """Get comprehensive risk metrics."""
    open_positions = await repo.get_open(db)
    closed_positions = await repo.get_closed(db, limit=1000)

    cb_status = circuit_breaker.get_status()

    total_exposure = sum(float(p.current_price) * float(p.quantity) for p in open_positions)
    returns = [float(p.realized_pnl) for p in closed_positions if p.realized_pnl]

    # Calculate basic risk metrics
    import numpy as np
    sharpe = 0.0
    sortino = 0.0
    if len(returns) > 1:
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret > 0:
            sharpe = (mean_ret / std_ret) * np.sqrt(252)
        downside = np.std([r for r in returns if r < 0]) or 1.0
        sortino = (mean_ret / downside) * np.sqrt(252)

    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    win_rate = len(wins) / len(returns) if returns else 0.0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses else 0.0

    return {
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "total_trades": len(returns),
        "open_positions": len(open_positions),
        "total_exposure": round(total_exposure, 2),
        "circuit_breaker": cb_status,
    }


@router.get("/risk-history")
async def get_risk_history(
    limit: int = 30,
    _user: dict = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
    repo: PositionRepository = Depends(get_position_repository),
):
    """Get historical risk metrics over time."""
    closed = await repo.get_closed(db, limit=limit * 10)

    # Build cumulative PnL series
    history = []
    cumulative_pnl = 0.0
    peak = 0.0

    for pos in reversed(closed):
        pnl = float(pos.realized_pnl) if pos.realized_pnl else 0.0
        cumulative_pnl += pnl
        peak = max(peak, cumulative_pnl)
        drawdown = (peak - cumulative_pnl) / peak if peak > 0 else 0.0

        history.append({
            "date": pos.closed_at.isoformat() if pos.closed_at else "",
            "cumulative_pnl": round(cumulative_pnl, 2),
            "drawdown_pct": round(drawdown, 4),
        })

    return history[-limit:]


@router.get("/fees")
async def get_portfolio_fees(
    period: str = Query(default="30d", pattern="^(7d|30d|90d|all)$"),
    _user: dict = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
    repo: PositionRepository = Depends(get_position_repository),
):
    """Get trading fees summary for a given period."""
    try:
        closed = await repo.get_closed(db, limit=10000)

        total_fees = 0.0
        fee_breakdown = []
        for pos in closed:
            fee = float(pos.fee) if hasattr(pos, "fee") and pos.fee else 0.0
            total_fees += fee

        return {
            "period": period,
            "total_fees": round(total_fees, 4),
            "currency": "USDT",
            "breakdown": fee_breakdown,
        }
    except Exception as e:
        logger.warning("fees_fetch_failed", error=str(e))
        return {
            "period": period,
            "total_fees": 0.0,
            "currency": "USDT",
            "breakdown": [],
        }


@router.get("/optimize")
async def get_portfolio_optimization(
    risk_tolerance: float = Query(default=0.5, ge=0.0, le=1.0),
    _user: dict = Depends(require_auth),
):
    """Get portfolio optimization suggestions based on risk tolerance."""
    try:
        from app.services.portfolio.optimizer import portfolio_optimizer
        result = await portfolio_optimizer.optimize(risk_tolerance=risk_tolerance)
        return result
    except Exception as e:
        logger.warning("optimization_failed", error=str(e))
        return {
            "risk_tolerance": risk_tolerance,
            "suggested_allocations": [],
            "expected_return": 0.0,
            "expected_volatility": 0.0,
            "status": "unavailable",
        }
