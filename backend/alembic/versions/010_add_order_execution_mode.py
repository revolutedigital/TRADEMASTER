"""Record the execution destination for every order.

Revision ID: 010
Revises: 009
Create Date: 2026-08-31
"""

import sqlalchemy as sa

from alembic import op

revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "orders",
        sa.Column("execution_mode", sa.String(length=10), nullable=False, server_default="PAPER"),
    )
    op.create_check_constraint(
        "ck_orders_execution_mode",
        "orders",
        "execution_mode IN ('PAPER', 'TESTNET', 'LIVE')",
    )
    op.create_index(
        "ix_orders_execution_mode_created_at",
        "orders",
        ["execution_mode", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_orders_execution_mode_created_at", table_name="orders")
    op.drop_constraint("ck_orders_execution_mode", "orders", type_="check")
    op.drop_column("orders", "execution_mode")
