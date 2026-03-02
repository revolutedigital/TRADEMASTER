"""Admin API endpoints: feature flags, deep health, system management."""
import time
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, require_auth
from app.core.feature_flags import feature_flags
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])

_START_TIME = time.time()


@router.get("/feature-flags")
async def list_feature_flags(_auth: dict = Depends(require_auth)):
    return feature_flags.list_flags()


@router.post("/feature-flags/{flag}/toggle")
async def toggle_feature_flag(flag: str, _auth: dict = Depends(require_auth)):
    new_state = feature_flags.toggle(flag)
    return {"flag": flag, "enabled": new_state}


@router.get("/health/deep")
async def deep_health_check(db: AsyncSession = Depends(get_db)):
    """Deep health check: verify all dependencies."""
    checks = {}

    # Database
    try:
        from sqlalchemy import text
        await db.execute(text("SELECT 1"))
        checks["database"] = {"status": "healthy", "latency_ms": 0}
    except Exception as e:
        checks["database"] = {"status": "unhealthy", "error": str(e)}

    # Redis
    try:
        from app.core.events import event_bus
        if event_bus._redis:
            await event_bus._redis.ping()
            checks["redis"] = {"status": "healthy"}
        else:
            checks["redis"] = {"status": "not_connected"}
    except Exception as e:
        checks["redis"] = {"status": "unhealthy", "error": str(e)}

    # Binance
    try:
        from app.services.exchange.binance_client import binance_client
        if binance_client._client:
            checks["binance"] = {"status": "connected", "circuit_breaker": binance_client._circuit_breaker.state}
        else:
            checks["binance"] = {"status": "not_connected"}
    except Exception as e:
        checks["binance"] = {"status": "unhealthy", "error": str(e)}

    overall = "healthy" if all(c.get("status") in ("healthy", "connected") for c in checks.values()) else "degraded"

    return {
        "status": overall,
        "uptime_seconds": round(time.time() - _START_TIME),
        "dependencies": checks,
    }
