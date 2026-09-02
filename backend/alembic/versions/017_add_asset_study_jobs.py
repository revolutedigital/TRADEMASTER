"""Persist asynchronous full-asset study jobs.

Revision ID: 017
Revises: 016
Create Date: 2026-09-01
"""

import sqlalchemy as sa

from alembic import op

revision = "017"
down_revision = "016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "asset_study_jobs",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("requested_by", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="QUEUED"),
        sa.Column("message", sa.Text()),
        sa.Column("study_json", sa.Text()),
        sa.Column("error_message", sa.Text()),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint(
            "status IN ('QUEUED', 'RUNNING', 'COMPLETED', 'FAILED', 'INTERRUPTED')",
            name="ck_asset_study_jobs_status",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.execute(
        "CREATE UNIQUE INDEX uq_asset_study_jobs_active "
        "ON asset_study_jobs ((1)) WHERE status IN ('QUEUED', 'RUNNING')"
    )
    op.create_index("ix_asset_study_jobs_symbol_created", "asset_study_jobs", ["symbol", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_asset_study_jobs_symbol_created", table_name="asset_study_jobs")
    op.execute("DROP INDEX uq_asset_study_jobs_active")
    op.drop_table("asset_study_jobs")
