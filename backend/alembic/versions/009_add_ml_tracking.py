"""Add ML tracking tables for experiment and prediction logging.

Revision ID: 009
Revises: 008
Create Date: 2026-03-15

Creates:
- ml_training_runs: Records each model training run with metrics + hyperparams
- ml_prediction_logs: Logs every prediction for rolling accuracy tracking
"""
from alembic import op
import sqlalchemy as sa

revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ml_training_runs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("model_type", sa.String(50), nullable=False, index=True),
        sa.Column("symbol", sa.String(20), nullable=False, index=True),
        sa.Column("metrics", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("hyperparams", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("dataset_hash", sa.String(64), nullable=True),
        sa.Column("dataset_size", sa.Integer(), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="completed"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "ml_prediction_logs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("model_type", sa.String(50), nullable=False, index=True),
        sa.Column("symbol", sa.String(20), nullable=False, index=True),
        sa.Column("signal", sa.String(10), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("features_hash", sa.String(64), nullable=True),
        sa.Column("actual_outcome", sa.String(10), nullable=True),
        sa.Column("outcome_pnl", sa.Float(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # Index for rolling accuracy queries (recent predictions by model+symbol)
    op.create_index(
        "ix_prediction_logs_lookup",
        "ml_prediction_logs",
        ["model_type", "symbol", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_prediction_logs_lookup", table_name="ml_prediction_logs")
    op.drop_table("ml_prediction_logs")
    op.drop_table("ml_training_runs")
