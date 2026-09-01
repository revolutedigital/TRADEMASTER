"""Persistent evidence for controlled exchange-execution release checks."""

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class ExecutionReleaseCheck(Base, TimestampMixin):
    """A non-secret audit record proving a bounded execution safety check passed."""

    __tablename__ = "execution_release_checks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    check_name: Mapped[str] = mapped_column(String(80), nullable=False)
    environment: Mapped[str] = mapped_column(String(10), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    entry_exchange_order_id: Mapped[str | None] = mapped_column(String(64))
    protective_order_list_id: Mapped[int | None] = mapped_column(BigInteger)
    exit_exchange_order_id: Mapped[str | None] = mapped_column(String(64))
    verified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index(
            "ix_execution_release_checks_lookup",
            "check_name",
            "environment",
            "status",
            "verified_at",
        ),
    )
