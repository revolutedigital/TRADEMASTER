"""Audit logging service for security-critical actions."""

import json
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import Request

from app.core.logging import get_logger
from app.models.audit import AuditLog

logger = get_logger(__name__)


class AuditLogger:
    """Records audit events for compliance and forensics."""

    async def log(
        self,
        db: AsyncSession,
        user_id: str,
        action: str,
        resource: str | None = None,
        details: dict | None = None,
        request: Request | None = None,
    ) -> AuditLog:
        ip_address = None
        user_agent = None
        if request:
            ip_address = request.client.host if request.client else None
            user_agent = request.headers.get("user-agent", "")[:500]

        entry = AuditLog(
            user_id=user_id,
            action=action,
            resource=resource,
            details=json.dumps(details) if details else None,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        db.add(entry)
        await db.flush()

        logger.info(
            "audit_event",
            user_id=user_id,
            action=action,
            resource=resource,
            ip=ip_address,
        )
        return entry

    async def get_logs(
        self,
        db: AsyncSession,
        user_id: str | None = None,
        action: str | None = None,
        limit: int = 100,
    ) -> list[AuditLog]:
        from sqlalchemy import select
        query = select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)
        if user_id:
            query = query.where(AuditLog.user_id == user_id)
        if action:
            query = query.where(AuditLog.action == action)
        result = await db.execute(query)
        return list(result.scalars().all())


audit_logger = AuditLogger()
