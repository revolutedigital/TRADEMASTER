"""Persist evidence-backed technical strategy deployments.

Revision ID: 015
Revises: 014
Create Date: 2026-08-31
"""

import sqlalchemy as sa

from alembic import op

revision = "015"
down_revision = "014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "strategy_deployments",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("source_backtest_id", sa.BigInteger(), nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("interval", sa.String(length=5), nullable=False),
        sa.Column("strategy_config_json", sa.Text(), nullable=False),
        sa.Column("execution_config_json", sa.Text(), nullable=False),
        sa.Column("target_execution_mode", sa.String(length=10), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("total_test_trades", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("walk_forward_windows", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("avg_return_pct", sa.Float(), nullable=False, server_default="0"),
        sa.Column("avg_sharpe", sa.Float(), nullable=False, server_default="0"),
        sa.Column("avg_max_drawdown_pct", sa.Float(), nullable=False, server_default="0"),
        sa.Column("avg_profit_factor", sa.Float(), nullable=False, server_default="0"),
        sa.Column("consistency_score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("overfitting_score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("rejection_reason", sa.Text()),
        sa.Column("activated_at", sa.DateTime(timezone=True)),
        sa.Column("deactivated_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["source_backtest_id"], ["backtest_results.id"], ondelete="RESTRICT"
        ),
        sa.CheckConstraint(
            "target_execution_mode IN ('PAPER', 'TESTNET', 'LIVE')",
            name="ck_strategy_deployments_execution_mode",
        ),
        sa.CheckConstraint(
            "status IN ('APPROVED', 'ACTIVE', 'REJECTED', 'DISABLED')",
            name="ck_strategy_deployments_status",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_strategy_deployments_runtime",
        "strategy_deployments",
        ["symbol", "interval", "target_execution_mode", "status"],
    )
    op.create_index(
        "uq_strategy_deployments_active_runtime",
        "strategy_deployments",
        ["symbol", "interval", "target_execution_mode"],
        unique=True,
        postgresql_where=sa.text("status = 'ACTIVE'"),
    )


def downgrade() -> None:
    op.drop_index("uq_strategy_deployments_active_runtime", table_name="strategy_deployments")
    op.drop_index("ix_strategy_deployments_runtime", table_name="strategy_deployments")
    op.drop_table("strategy_deployments")
