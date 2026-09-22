"""Add explicit research-only Testnet release records.

Revision ID: 022
Revises: 021
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "022"
down_revision: str | None = "021"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_identity = sa.BigInteger().with_variant(sa.Integer(), "sqlite")
_now = sa.text("CURRENT_TIMESTAMP")


def upgrade() -> None:
    op.create_table(
        "research_testnet_releases",
        sa.Column("id", _identity, primary_key=True, autoincrement=True),
        sa.Column("experiment_id", sa.String(length=36), nullable=False, unique=True),
        sa.Column("release_sha256", sa.String(length=64), nullable=False, unique=True),
        sa.Column("requested_by", sa.String(length=120), nullable=False),
        sa.Column("reasons_json", sa.Text(), nullable=False),
        sa.Column("evidence_snapshot_json", sa.Text(), nullable=False),
        sa.Column("released_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=_now),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=_now),
        sa.ForeignKeyConstraint(
            ["experiment_id"], ["research_experiments.id"], ondelete="RESTRICT"
        ),
    )
    op.create_index(
        "ix_research_testnet_releases_lookup",
        "research_testnet_releases",
        ["experiment_id", "released_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_research_testnet_releases_lookup",
        table_name="research_testnet_releases",
    )
    op.drop_table("research_testnet_releases")
