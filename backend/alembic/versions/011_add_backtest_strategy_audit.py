"""Persist the exact strategy and complete metrics for a backtest.

Revision ID: 011
Revises: 010
Create Date: 2026-08-31
"""

import sqlalchemy as sa

from alembic import op

revision = "011"
down_revision = "010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "backtest_results",
        sa.Column(
            "strategy_name",
            sa.String(length=100),
            nullable=False,
            server_default="ML ensemble",
        ),
    )
    op.add_column(
        "backtest_results",
        sa.Column(
            "execution_profile",
            sa.String(length=30),
            nullable=False,
            server_default="model_long_short",
        ),
    )
    op.add_column(
        "backtest_results",
        sa.Column("strategy_config_json", sa.Text(), nullable=False, server_default="{}"),
    )
    op.add_column(
        "backtest_results",
        sa.Column("total_return", sa.Float(), nullable=False, server_default="0"),
    )
    op.add_column(
        "backtest_results",
        sa.Column("winning_trades", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "backtest_results",
        sa.Column("losing_trades", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "backtest_results",
        sa.Column("max_drawdown", sa.Float(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("backtest_results", "max_drawdown")
    op.drop_column("backtest_results", "losing_trades")
    op.drop_column("backtest_results", "winning_trades")
    op.drop_column("backtest_results", "total_return")
    op.drop_column("backtest_results", "strategy_config_json")
    op.drop_column("backtest_results", "execution_profile")
    op.drop_column("backtest_results", "strategy_name")
