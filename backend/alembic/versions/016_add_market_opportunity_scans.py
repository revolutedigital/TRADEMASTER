"""Persist asynchronous market-wide opportunity scans.

Revision ID: 016
Revises: 015
Create Date: 2026-09-01
"""

import sqlalchemy as sa

from alembic import op

revision = "016"
down_revision = "015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "market_opportunity_scans",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="QUEUED"),
        sa.Column("requested_by", sa.String(length=128), nullable=False),
        sa.Column("total_assets", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("screened_assets", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("shortlisted_assets", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("studied_assets", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed_assets", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("message", sa.Text()),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint(
            "status IN ('QUEUED', 'RUNNING', 'COMPLETED', 'FAILED', 'INTERRUPTED')",
            name="ck_market_opportunity_scans_status",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.execute(
        "CREATE UNIQUE INDEX uq_market_opportunity_scans_active "
        "ON market_opportunity_scans ((1)) "
        "WHERE status IN ('QUEUED', 'RUNNING')"
    )

    op.create_table(
        "market_opportunity_candidates",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("scan_id", sa.BigInteger(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("screening_score", sa.Float(), nullable=False),
        sa.Column("market_trend", sa.String(length=10), nullable=False),
        sa.Column("price_change_pct_24h", sa.Float(), nullable=False),
        sa.Column("quote_volume_24h", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="SHORTLISTED"),
        sa.Column("study_json", sa.Text()),
        sa.Column("error_message", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint(
            "market_trend IN ('UPTREND', 'DOWNTREND', 'RANGE')",
            name="ck_market_opportunity_candidates_trend",
        ),
        sa.CheckConstraint(
            "status IN ('SHORTLISTED', 'STUDYING', 'APPROVED', 'REJECTED', 'UNAVAILABLE', 'FAILED')",
            name="ck_market_opportunity_candidates_status",
        ),
        sa.ForeignKeyConstraint(
            ["scan_id"], ["market_opportunity_scans.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_market_opportunity_candidates_scan_rank",
        "market_opportunity_candidates",
        ["scan_id", "rank"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_market_opportunity_candidates_scan_rank", table_name="market_opportunity_candidates")
    op.drop_table("market_opportunity_candidates")
    op.execute("DROP INDEX uq_market_opportunity_scans_active")
    op.drop_table("market_opportunity_scans")
