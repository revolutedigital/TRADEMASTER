"""Audit log API endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, require_auth
from app.core.audit import audit_logger

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("/logs")
async def get_audit_logs(
    user_id: str | None = None,
    action: str | None = None,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    _auth: dict = Depends(require_auth),
):
    logs = await audit_logger.get_logs(db, user_id=user_id, action=action, limit=min(limit, 500))
    return [
        {
            "id": log.id,
            "user_id": log.user_id,
            "action": log.action,
            "resource": log.resource,
            "details": log.details,
            "ip_address": log.ip_address,
            "created_at": str(log.created_at),
        }
        for log in logs
    ]
