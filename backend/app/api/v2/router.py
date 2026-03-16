"""API v2 router with improved response formats, versioning, and deprecation headers.

V2 wraps selected V1 endpoints with:
- Consistent envelope: {"data": ..., "meta": {"api_version": "2.0", ...}}
- Deprecation header on responses that signal V1 sunset timeline
"""

from fastapi import APIRouter, Response
from starlette.requests import Request

v2_router = APIRouter()


def _add_deprecation_headers(response: Response, sunset: str = "2027-01-01") -> None:
    """Add standard deprecation and sunset headers to a response."""
    response.headers["Deprecation"] = "false"
    response.headers["Sunset"] = sunset
    response.headers["X-API-Version"] = "2.0"


@v2_router.get("/health")
async def health(response: Response):
    """V2 health endpoint with API version metadata."""
    _add_deprecation_headers(response)
    return {"status": "healthy", "api_version": "2.0"}


@v2_router.get("/health/detailed")
async def health_detailed(response: Response):
    """V2 proxy to the detailed health check with v2 envelope."""
    from app.api.v1.system import health_detailed as v1_health_detailed

    _add_deprecation_headers(response)
    data = await v1_health_detailed()
    return {
        "data": data,
        "meta": {"api_version": "2.0"},
    }


@v2_router.get("/portfolio/summary")
async def portfolio_summary(request: Request, response: Response):
    """V2 portfolio summary with enriched response format."""
    from app.queries.get_portfolio import get_portfolio_query
    from app.models.base import async_session

    _add_deprecation_headers(response)
    async with async_session() as db:
        data = await get_portfolio_query.execute(db)
    return {
        "data": data,
        "meta": {"api_version": "2.0", "deprecation": None},
    }


@v2_router.get("/risk/metrics")
async def risk_metrics(response: Response):
    """V2 risk metrics with enriched response."""
    from app.queries.get_risk_metrics import get_risk_metrics_query

    _add_deprecation_headers(response)
    data = await get_risk_metrics_query.execute()
    return {
        "data": data,
        "meta": {"api_version": "2.0"},
    }


@v2_router.get("/trades")
async def trade_history(
    response: Response,
    symbol: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    """V2 trade history with pagination metadata."""
    from app.queries.get_trade_history import get_trade_history_query
    from app.models.base import async_session

    _add_deprecation_headers(response)
    async with async_session() as db:
        trades = await get_trade_history_query.execute(
            db, symbol=symbol, limit=limit, offset=offset
        )
    return {
        "data": trades,
        "meta": {
            "api_version": "2.0",
            "limit": limit,
            "offset": offset,
            "count": len(trades),
        },
    }
