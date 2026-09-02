"""Durable full-asset research jobs that never hold an HTTP request open."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, CheckConstraint, DateTime, Index, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class AssetStudyJob(Base, TimestampMixin):
    """One safe asynchronous study; its result still cannot activate trading."""

    __tablename__ = "asset_study_jobs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    requested_by: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="QUEUED")
    message: Mapped[str | None] = mapped_column(Text)
    study_json: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        # One study at a time avoids duplicate history/model/backtest writes
        # and keeps the Railway API responsive while a full study runs.
        Index(
            "uq_asset_study_jobs_active",
            text("(1)"),
            unique=True,
            postgresql_where=text("status IN ('QUEUED', 'RUNNING')"),
        ),
        Index("ix_asset_study_jobs_symbol_created", "symbol", "created_at"),
        CheckConstraint(
            "status IN ('QUEUED', 'RUNNING', 'COMPLETED', 'FAILED', 'INTERRUPTED')",
            name="ck_asset_study_jobs_status",
        ),
    )
