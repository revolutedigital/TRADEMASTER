"""Audit logging for security-critical operations."""

from datetime import datetime, timezone
from app.core.logging import get_logger

logger = get_logger(__name__)


class AuditLogger:
    """Records security-relevant events for compliance and forensics.

    Events are logged to structured logging (which goes to stdout/Railway logs)
    and can optionally be persisted to the database.
    """

    async def log_event(
        self,
        action: str,
        user_id: str = "system",
        resource: str | None = None,
        details: dict | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> None:
        """Record an audit event.

        Args:
            action: Event type (LOGIN, LOGOUT, TRADE, CONFIG_CHANGE, EXPORT, etc.)
            user_id: Who performed the action
            resource: What was affected (e.g., "order:123", "settings:risk")
            details: Additional context (before/after values, etc.)
            ip_address: Client IP
            user_agent: Client user agent string
        """
        event = {
            "audit_action": action,
            "audit_user": user_id,
            "audit_resource": resource,
            "audit_details": details,
            "audit_ip": ip_address,
            "audit_ua": user_agent,
            "audit_ts": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("audit_event", **{k: v for k, v in event.items() if v is not None})

        # Persist to database if available
        try:
            from app.models.base import async_session_factory
            from app.models.audit import AuditLog
            async with async_session_factory() as db:
                log_entry = AuditLog(
                    user_id=user_id,
                    action=action,
                    resource=resource,
                    details=str(details) if details else None,
                    ip_address=ip_address,
                    user_agent=user_agent,
                )
                db.add(log_entry)
                await db.commit()
        except Exception as e:
            # Never let audit logging break the application
            logger.warning("audit_db_persist_failed", error=str(e))

    async def log_login(self, username: str, success: bool, ip: str | None = None) -> None:
        await self.log_event(
            action="LOGIN_SUCCESS" if success else "LOGIN_FAILED",
            user_id=username,
            ip_address=ip,
        )

    async def log_trade(self, user_id: str, order_id: int, symbol: str, side: str, quantity: float) -> None:
        await self.log_event(
            action="TRADE_EXECUTED",
            user_id=user_id,
            resource=f"order:{order_id}",
            details={"symbol": symbol, "side": side, "quantity": quantity},
        )

    async def log_config_change(self, user_id: str, setting: str, old_value: str, new_value: str) -> None:
        await self.log_event(
            action="CONFIG_CHANGE",
            user_id=user_id,
            resource=f"settings:{setting}",
            details={"old": old_value, "new": new_value},
        )

audit_logger = AuditLogger()
