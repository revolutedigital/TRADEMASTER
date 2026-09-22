"""Immutable experiment and data-use ledger for microstructure research."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Boolean,
    Float,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


_IDENTITY = BigInteger().with_variant(Integer(), "sqlite")


class ResearchExperiment(Base, TimestampMixin):
    """One preregistered experiment; only drafts may change their definition."""

    __tablename__ = "research_experiments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="DRAFT")
    code_revision: Mapped[str] = mapped_column(String(40), nullable=False)
    protocol_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    product_json: Mapped[str] = mapped_column(Text, nullable=False)
    cost_profile_json: Mapped[str] = mapped_column(Text, nullable=False)
    approval_gate_json: Mapped[str] = mapped_column(Text, nullable=False)
    experiment_sha256: Mapped[str | None] = mapped_column(String(64), unique=True)
    decision_reasons_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    frozen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint(
            "status IN ('DRAFT', 'FROZEN', 'REJECTED', 'INCONCLUSIVE', 'APPROVED')",
            name="ck_research_experiments_status",
        ),
        Index("ix_research_experiments_status_created", "status", "created_at"),
    )


class ResearchDataUse(Base):
    """A partition assigned to one temporal role, including when it was opened."""

    __tablename__ = "research_data_uses"

    id: Mapped[int] = mapped_column(_IDENTITY, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("research_experiments.id", ondelete="RESTRICT"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(24), nullable=False)
    start_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    manifest_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint(
            "role IN ('DEVELOPMENT', 'TRAINING', 'SELECTION', 'AUDIT', 'PROSPECTIVE_SHADOW')",
            name="ck_research_data_uses_role",
        ),
        CheckConstraint("end_at > start_at", name="ck_research_data_uses_window"),
        UniqueConstraint("experiment_id", "role", name="uq_research_data_use_role"),
        Index(
            "ix_research_data_uses_manifest",
            "manifest_sha256",
            "role",
            "opened_at",
        ),
    )


class ResearchHypothesisAttempt(Base):
    """One globally countable hypothesis attempt, including failed evaluations."""

    __tablename__ = "research_hypothesis_attempts"

    id: Mapped[int] = mapped_column(_IDENTITY, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("research_experiments.id", ondelete="RESTRICT"),
        nullable=False,
    )
    kind: Mapped[str] = mapped_column(String(40), nullable=False)
    fingerprint_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    definition_json: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="REGISTERED")
    outcome_opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint(
            "status IN ('REGISTERED', 'EVALUATED', 'FAILED')",
            name="ck_research_hypothesis_attempts_status",
        ),
        UniqueConstraint(
            "experiment_id",
            "fingerprint_sha256",
            name="uq_research_hypothesis_attempt",
        ),
        Index("ix_research_hypotheses_fingerprint", "fingerprint_sha256"),
    )


class ResearchExperimentEvent(Base):
    """Append-only audit event for experiment lifecycle changes."""

    __tablename__ = "research_experiment_events"

    id: Mapped[int] = mapped_column(_IDENTITY, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("research_experiments.id", ondelete="RESTRICT"),
        nullable=False,
    )
    kind: Mapped[str] = mapped_column(String(40), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_research_experiment_events_timeline", "experiment_id", "occurred_at"),
    )


class ResearchShadowSignal(Base):
    """Append-only hypothetical signal; this model has no execution fields."""

    __tablename__ = "research_shadow_signals"

    id: Mapped[int] = mapped_column(_IDENTITY, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("research_experiments.id", ondelete="RESTRICT"),
        nullable=False,
    )
    decision_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    side: Mapped[str] = mapped_column(String(4), nullable=False)
    horizon_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    would_enter: Mapped[bool] = mapped_column(Boolean, nullable=False)
    model_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_vector_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    outcome_json: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint("side IN ('BUY', 'SELL')", name="ck_research_shadow_side"),
        CheckConstraint("horizon_seconds IN (120, 300)", name="ck_research_shadow_horizon"),
        CheckConstraint(
            "probability >= 0 AND probability <= 1",
            name="ck_research_shadow_probability",
        ),
        CheckConstraint("threshold >= 0 AND threshold <= 1", name="ck_research_shadow_threshold"),
        UniqueConstraint(
            "experiment_id",
            "decision_time",
            "side",
            "horizon_seconds",
            name="uq_research_shadow_signal",
        ),
        Index("ix_research_shadow_experiment_time", "experiment_id", "decision_time"),
    )
