"""Persisted, non-executing market-wide opportunity scans."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class MarketOpportunityScan(Base, TimestampMixin):
    """One asynchronous scan across the current liquid Spot catalog."""

    __tablename__ = "market_opportunity_scans"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="QUEUED")
    requested_by: Mapped[str] = mapped_column(String(128), nullable=False)
    total_assets: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    screened_assets: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    shortlisted_assets: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    studied_assets: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_assets: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    candidates: Mapped[list[MarketOpportunityCandidate]] = relationship(
        back_populates="scan",
        cascade="all, delete-orphan",
        order_by="MarketOpportunityCandidate.rank",
    )

    __table_args__ = (
        # A partial unique index on a constant is a PostgreSQL-safe singleton
        # lock across QUEUED and RUNNING states, including concurrent requests.
        Index(
            "uq_market_opportunity_scans_active",
            text("(1)"),
            unique=True,
            postgresql_where=text("status IN ('QUEUED', 'RUNNING')"),
        ),
        CheckConstraint(
            "status IN ('QUEUED', 'RUNNING', 'COMPLETED', 'FAILED', 'INTERRUPTED')",
            name="ck_market_opportunity_scans_status",
        ),
    )


class MarketOpportunityCandidate(Base, TimestampMixin):
    """A finalist from a market-wide screen and its full-study outcome."""

    __tablename__ = "market_opportunity_candidates"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    scan_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("market_opportunity_scans.id", ondelete="CASCADE"),
        nullable=False,
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    screening_score: Mapped[float] = mapped_column(Float, nullable=False)
    market_trend: Mapped[str] = mapped_column(String(10), nullable=False)
    price_change_pct_24h: Mapped[float] = mapped_column(Float, nullable=False)
    quote_volume_24h: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="SHORTLISTED")
    study_json: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    scan: Mapped[MarketOpportunityScan] = relationship(back_populates="candidates")

    __table_args__ = (
        Index("ix_market_opportunity_candidates_scan_rank", "scan_id", "rank", unique=True),
        CheckConstraint(
            "market_trend IN ('UPTREND', 'DOWNTREND', 'RANGE')",
            name="ck_market_opportunity_candidates_trend",
        ),
        CheckConstraint(
            "status IN ('SHORTLISTED', 'STUDYING', 'APPROVED', 'REJECTED', 'UNAVAILABLE', 'FAILED')",
            name="ck_market_opportunity_candidates_status",
        ),
    )
