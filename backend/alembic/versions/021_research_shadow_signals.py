"""Add append-only prospective research shadow signals.

Revision ID: 021
Revises: 020
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "021"
down_revision: str | None = "020"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "research_shadow_signals",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("experiment_id", sa.String(length=36), nullable=False),
        sa.Column("decision_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("side", sa.String(length=4), nullable=False),
        sa.Column("horizon_seconds", sa.Integer(), nullable=False),
        sa.Column("probability", sa.Float(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("would_enter", sa.Boolean(), nullable=False),
        sa.Column("model_sha256", sa.String(length=64), nullable=False),
        sa.Column("feature_vector_sha256", sa.String(length=64), nullable=False),
        sa.Column("outcome_json", sa.Text(), nullable=True),
        sa.CheckConstraint("side IN ('BUY', 'SELL')", name="ck_research_shadow_side"),
        sa.CheckConstraint("horizon_seconds IN (120, 300)", name="ck_research_shadow_horizon"),
        sa.CheckConstraint(
            "probability >= 0 AND probability <= 1",
            name="ck_research_shadow_probability",
        ),
        sa.CheckConstraint(
            "threshold >= 0 AND threshold <= 1", name="ck_research_shadow_threshold"
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"], ["research_experiments.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "experiment_id",
            "decision_time",
            "side",
            "horizon_seconds",
            name="uq_research_shadow_signal",
        ),
    )
    op.create_index(
        "ix_research_shadow_experiment_time",
        "research_shadow_signals",
        ["experiment_id", "decision_time"],
    )


def downgrade() -> None:
    op.drop_index("ix_research_shadow_experiment_time", table_name="research_shadow_signals")
    op.drop_table("research_shadow_signals")
