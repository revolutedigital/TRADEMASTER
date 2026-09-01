"""Persist native protection state for Binance Spot positions.

Revision ID: 012
Revises: 011
Create Date: 2026-08-31
"""

import sqlalchemy as sa
from alembic import op


revision = "012"
down_revision = "011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("orders", sa.Column("protective_order_list_id", sa.BigInteger()))
    op.add_column(
        "positions",
        sa.Column("execution_mode", sa.String(length=10), nullable=False, server_default="PAPER"),
    )
    op.create_check_constraint(
        "ck_positions_execution_mode",
        "positions",
        "execution_mode IN ('PAPER', 'TESTNET', 'LIVE')",
    )
    op.add_column("positions", sa.Column("entry_exchange_order_id", sa.String(length=64)))
    op.add_column("positions", sa.Column("protective_order_list_id", sa.BigInteger()))
    op.add_column(
        "positions",
        sa.Column("protection_status", sa.String(length=20), nullable=False, server_default="LOCAL"),
    )
    op.add_column("positions", sa.Column("protection_updated_at", sa.DateTime(timezone=True)))
    op.create_index(
        "ix_positions_live_protection",
        "positions",
        ["execution_mode", "protection_status", "is_open"],
    )


def downgrade() -> None:
    op.drop_index("ix_positions_live_protection", table_name="positions")
    op.drop_column("positions", "protection_updated_at")
    op.drop_column("positions", "protection_status")
    op.drop_column("positions", "protective_order_list_id")
    op.drop_column("positions", "entry_exchange_order_id")
    op.drop_constraint("ck_positions_execution_mode", "positions", type_="check")
    op.drop_column("positions", "execution_mode")
    op.drop_column("orders", "protective_order_list_id")
