"""User model for multi-tenant support."""

from sqlalchemy import Column, String, Boolean, DateTime, Text, func, BigInteger
from app.models.base import Base


class User(Base):
    """Application user with role-based access."""

    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    role = Column(String(20), nullable=False, default="trader")  # admin, trader, viewer
    is_active = Column(Boolean, default=True, nullable=False)
    last_login = Column(DateTime(timezone=True))
    preferences = Column(Text)  # JSON preferences

    # TOTP two-factor authentication
    totp_enabled = Column(Boolean, default=False, nullable=False, server_default="false")
    totp_secret = Column(String(64), nullable=True)  # Base32-encoded shared secret
    totp_backup_codes = Column(Text, nullable=True)  # JSON array of backup codes

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
