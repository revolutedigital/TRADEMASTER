"""Admin API endpoints: feature flags, deep health, system management."""
import time
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, require_auth
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])

_START_TIME = time.time()


@router.get("/feature-flags")
async def get_feature_flags(_user: dict = Depends(require_auth)):
    from app.core.feature_flags import feature_flags
    return {"flags": feature_flags.get_all()}


@router.put("/feature-flags/{flag_name}")
async def toggle_feature_flag(flag_name: str, enabled: bool = True, _user: dict = Depends(require_auth)):
    from app.core.feature_flags import feature_flags
    feature_flags.set_flag(flag_name, enabled)
    return {"flag": flag_name, "enabled": enabled}


@router.get("/audit/logs")
async def get_audit_logs(
    limit: int = 50,
    action: str | None = None,
    _user: dict = Depends(require_auth),
):
    """Get recent audit log entries."""
    from app.models.base import async_session_factory
    from app.models.audit import AuditLog
    from sqlalchemy import select

    async with async_session_factory() as db:
        query = select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)
        if action:
            query = query.where(AuditLog.action == action)
        result = await db.execute(query)
        logs = result.scalars().all()

    return {
        "count": len(logs),
        "logs": [
            {
                "id": log.id,
                "user_id": log.user_id,
                "action": log.action,
                "resource": log.resource,
                "details": log.details,
                "ip_address": log.ip_address,
                "created_at": log.created_at.isoformat() if log.created_at else None,
            }
            for log in logs
        ],
    }


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
