"""Add stored_events table and TOTP columns to users.

Revision ID: 007
Revises: 006
Create Date: 2026-03-15
"""
from alembic import op
import sqlalchemy as sa

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- GAP 1: Persistent event store ---
    op.create_table(
        "stored_events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("event_id", sa.String(36), unique=True, nullable=False, index=True),
        sa.Column("event_type", sa.String(100), nullable=False, index=True),
        sa.Column("aggregate_type", sa.String(100), nullable=False, index=True),
        sa.Column("aggregate_id", sa.String(200), nullable=False, index=True),
        sa.Column("data", sa.Text(), nullable=False),
        sa.Column("metadata", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("event_timestamp", sa.String(50), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # --- GAP 3: TOTP columns on users table ---
    op.add_column("users", sa.Column("totp_enabled", sa.Boolean(), server_default="false", nullable=False))
    op.add_column("users", sa.Column("totp_secret", sa.String(64), nullable=True))
    op.add_column("users", sa.Column("totp_backup_codes", sa.Text(), nullable=True))  # JSON array


def downgrade() -> None:
    op.drop_column("users", "totp_backup_codes")
    op.drop_column("users", "totp_secret")
    op.drop_column("users", "totp_enabled")
    op.drop_table("stored_events")
