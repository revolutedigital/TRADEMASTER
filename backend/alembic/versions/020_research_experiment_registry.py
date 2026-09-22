"""Add immutable microstructure research experiment registry.

Revision ID: 020
Revises: 019
Create Date: 2026-09-21

Rollback: ``alembic downgrade 019`` drops only the four research ledger tables.
"""

import sqlalchemy as sa

from alembic import op

revision = "020"
down_revision = "019"
branch_labels = None
depends_on = None

_identity = sa.BigInteger().with_variant(sa.Integer(), "sqlite")
_now = sa.text("CURRENT_TIMESTAMP")


def upgrade() -> None:
    op.create_table(
        "research_experiments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="DRAFT"),
        sa.Column("code_revision", sa.String(40), nullable=False),
        sa.Column("protocol_sha256", sa.String(64), nullable=False),
        sa.Column("product_json", sa.Text(), nullable=False),
        sa.Column("cost_profile_json", sa.Text(), nullable=False),
        sa.Column("approval_gate_json", sa.Text(), nullable=False),
        sa.Column("experiment_sha256", sa.String(64), unique=True),
        sa.Column("decision_reasons_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("frozen_at", sa.DateTime(timezone=True)),
        sa.Column("decided_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=_now),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=_now),
        sa.CheckConstraint(
            "status IN ('DRAFT', 'FROZEN', 'REJECTED', 'INCONCLUSIVE', 'APPROVED')",
            name="ck_research_experiments_status",
        ),
    )
    op.create_index(
        "ix_research_experiments_status_created",
        "research_experiments",
        ["status", "created_at"],
    )
    op.create_table(
        "research_data_uses",
        sa.Column("id", _identity, primary_key=True, autoincrement=True),
        sa.Column(
            "experiment_id",
            sa.String(36),
            sa.ForeignKey("research_experiments.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("role", sa.String(24), nullable=False),
        sa.Column("start_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("manifest_sha256", sa.String(64), nullable=False),
        sa.Column("opened_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "role IN ('DEVELOPMENT', 'TRAINING', 'SELECTION', 'AUDIT', 'PROSPECTIVE_SHADOW')",
            name="ck_research_data_uses_role",
        ),
        sa.CheckConstraint("end_at > start_at", name="ck_research_data_uses_window"),
        sa.UniqueConstraint("experiment_id", "role", name="uq_research_data_use_role"),
    )
    op.create_index(
        "ix_research_data_uses_manifest",
        "research_data_uses",
        ["manifest_sha256", "role", "opened_at"],
    )
    op.create_table(
        "research_hypothesis_attempts",
        sa.Column("id", _identity, primary_key=True, autoincrement=True),
        sa.Column(
            "experiment_id",
            sa.String(36),
            sa.ForeignKey("research_experiments.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(40), nullable=False),
        sa.Column("fingerprint_sha256", sa.String(64), nullable=False),
        sa.Column("definition_json", sa.Text(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="REGISTERED"),
        sa.Column("outcome_opened_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "status IN ('REGISTERED', 'EVALUATED', 'FAILED')",
            name="ck_research_hypothesis_attempts_status",
        ),
        sa.UniqueConstraint(
            "experiment_id",
            "fingerprint_sha256",
            name="uq_research_hypothesis_attempt",
        ),
    )
    op.create_index(
        "ix_research_hypotheses_fingerprint",
        "research_hypothesis_attempts",
        ["fingerprint_sha256"],
    )
    op.create_table(
        "research_experiment_events",
        sa.Column("id", _identity, primary_key=True, autoincrement=True),
        sa.Column(
            "experiment_id",
            sa.String(36),
            sa.ForeignKey("research_experiments.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(40), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_research_experiment_events_timeline",
        "research_experiment_events",
        ["experiment_id", "occurred_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_research_experiment_events_timeline",
        table_name="research_experiment_events",
    )
    op.drop_table("research_experiment_events")
    op.drop_index(
        "ix_research_hypotheses_fingerprint",
        table_name="research_hypothesis_attempts",
    )
    op.drop_table("research_hypothesis_attempts")
    op.drop_index("ix_research_data_uses_manifest", table_name="research_data_uses")
    op.drop_table("research_data_uses")
    op.drop_index(
        "ix_research_experiments_status_created",
        table_name="research_experiments",
    )
    op.drop_table("research_experiments")
