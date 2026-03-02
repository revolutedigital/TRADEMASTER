"""Add model metadata fields and backtest results table.

Revision ID: 003_model_fields
Revises: 002_timescaledb
Create Date: 2026-02-28
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "003_model_fields"
down_revision = "002_timescaledb"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add new columns to model_metadata
    op.add_column("model_metadata", sa.Column("training_metrics", sa.Text(), nullable=True))
    op.add_column("model_metadata", sa.Column("hyperparameters", sa.Text(), nullable=True))
    op.add_column("model_metadata", sa.Column("data_version", sa.String(50), nullable=True))

    # Create backtest_results table
    op.create_table(
        "backtest_results",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("interval", sa.String(5), nullable=False),
        sa.Column("initial_capital", sa.Numeric(20, 2), nullable=False),
        sa.Column("signal_threshold", sa.Numeric(5, 4), nullable=False),
        sa.Column("atr_stop_multiplier", sa.Numeric(5, 2), nullable=True),
        sa.Column("risk_reward_ratio", sa.Numeric(5, 2), nullable=True),
        sa.Column("total_trades", sa.Integer(), nullable=False),
        sa.Column("win_rate", sa.Float(), nullable=False),
        sa.Column("total_return_pct", sa.Float(), nullable=False),
        sa.Column("sharpe_ratio", sa.Float(), nullable=False),
        sa.Column("max_drawdown_pct", sa.Float(), nullable=False),
        sa.Column("profit_factor", sa.Float(), nullable=False),
        sa.Column("expectancy", sa.Float(), nullable=True),
        sa.Column("equity_curve_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create notifications table
    op.create_table(
        "notifications",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("type", sa.String(20), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("severity", sa.String(20), server_default="info", nullable=False),
        sa.Column("is_read", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("read_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    # Add optimized indexes
    op.create_index("ix_ohlcv_symbol_interval_time", "ohlcv", ["symbol", "interval", "open_time"])
    op.create_index("ix_notifications_unread", "notifications", ["is_read"], postgresql_where=sa.text("is_read = false"))


def downgrade() -> None:
    op.drop_index("ix_notifications_unread")
    op.drop_index("ix_ohlcv_symbol_interval_time")
    op.drop_table("notifications")
    op.drop_table("backtest_results")
    op.drop_column("model_metadata", "data_version")
    op.drop_column("model_metadata", "hyperparameters")
    op.drop_column("model_metadata", "training_metrics")
