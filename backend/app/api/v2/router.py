"""API v2 router with improved response formats and versioning."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from starlette.requests import Request

v2_router = APIRouter()


@v2_router.get("/health")
async def health():
    return {"status": "healthy", "api_version": "v2"}


@v2_router.get("/portfolio/summary")
async def portfolio_summary(request: Request):
    """V2 portfolio summary with enriched response format."""
    from app.queries.get_portfolio import get_portfolio_query
    from app.models.base import async_session
    async with async_session() as db:
        data = await get_portfolio_query.execute(db)
    return {
        "data": data,
        "meta": {"api_version": "v2", "deprecation": None},
    }


@v2_router.get("/risk/metrics")
async def risk_metrics():
    """V2 risk metrics with enriched response."""
    from app.queries.get_risk_metrics import get_risk_metrics_query
    data = await get_risk_metrics_query.execute()
    return {
        "data": data,
        "meta": {"api_version": "v2"},
    }


@v2_router.get("/trades")
async def trade_history(symbol: str | None = None, limit: int = 50, offset: int = 0):
    """V2 trade history with pagination metadata."""
    from app.queries.get_trade_history import get_trade_history_query
    from app.models.base import async_session
    async with async_session() as db:
        trades = await get_trade_history_query.execute(db, symbol=symbol, limit=limit, offset=offset)
    return {
        "data": trades,
        "meta": {"api_version": "v2", "limit": limit, "offset": offset, "count": len(trades)},
    }
