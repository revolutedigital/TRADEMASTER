"""Portfolio API endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db
from app.models.portfolio import Position
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
):
    """Get positions."""
    query = select(Position).where(Position.is_open == is_open).order_by(Position.created_at.desc()).limit(limit)
    if symbol:
        query = query.where(Position.symbol == symbol.upper())
    result = await db.execute(query)
    return list(result.scalars().all())


@router.get("/summary", response_model=PortfolioSummary)
async def get_portfolio_summary(db: AsyncSession = Depends(get_db)):
    """Get portfolio summary with all open positions."""
    result = await db.execute(
        select(Position).where(Position.is_open == True)
    )
    open_positions = list(result.scalars().all())

    total_unrealized = sum(float(p.unrealized_pnl) for p in open_positions)
    total_exposure = sum(float(p.current_price) * float(p.quantity) for p in open_positions)

    # Get total realized P&L from closed positions
    closed_result = await db.execute(
        select(Position).where(Position.is_open == False)
    )
    closed_positions = list(closed_result.scalars().all())
    total_realized = sum(float(p.realized_pnl) for p in closed_positions)

    # Query live balance from Binance
    total_equity = 0.0
    available_balance = 0.0
    try:
        from app.services.exchange.binance_client import binance_client
        balance = await binance_client.get_balance("USDT")
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
async def get_risk_status():
    """Get current risk management status."""
    return circuit_breaker.get_status()
