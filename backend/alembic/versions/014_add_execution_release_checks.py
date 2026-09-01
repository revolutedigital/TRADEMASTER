"""Persist controlled exchange-execution release evidence.

Revision ID: 014
Revises: 013
Create Date: 2026-08-31
"""

import sqlalchemy as sa

from alembic import op

revision = "014"
down_revision = "013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "execution_release_checks",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("check_name", sa.String(length=80), nullable=False),
        sa.Column("environment", sa.String(length=10), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("entry_exchange_order_id", sa.String(length=64)),
        sa.Column("protective_order_list_id", sa.BigInteger()),
        sa.Column("exit_exchange_order_id", sa.String(length=64)),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_execution_release_checks_lookup",
        "execution_release_checks",
        ["check_name", "environment", "status", "verified_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_execution_release_checks_lookup", table_name="execution_release_checks")
    op.drop_table("execution_release_checks")
