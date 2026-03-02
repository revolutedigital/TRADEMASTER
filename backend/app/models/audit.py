"""Audit log model for security and compliance tracking."""

from sqlalchemy import BigInteger, Column, String, Text, DateTime
from datetime import datetime, timezone

from app.models.base import Base


class AuditLog(Base):
    """Immutable audit trail for security-relevant actions."""
    __tablename__ = "audit_log"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(String(100), nullable=False, index=True)
    action = Column(String(50), nullable=False, index=True)  # LOGIN, LOGOUT, TRADE, CONFIG_CHANGE, EXPORT
    resource = Column(String(100))  # e.g., "order:123", "settings:risk"
    details = Column(Text)  # JSON with before/after or extra context
    ip_address = Column(String(45))  # IPv4 or IPv6
    user_agent = Column(String(500))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
