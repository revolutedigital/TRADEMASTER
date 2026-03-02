"""API Key model for programmatic access."""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text
from app.models.base import Base


class APIKey(Base):
    """API keys for programmatic access with scoped permissions."""
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), nullable=False, index=True)
    key_hash = Column(String(128), nullable=False, unique=True)
    key_prefix = Column(String(8), nullable=False)  # First 8 chars for identification
    name = Column(String(100), nullable=False)
    scopes = Column(Text, default="read")  # Comma-separated: read,trade,admin
    last_used_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
