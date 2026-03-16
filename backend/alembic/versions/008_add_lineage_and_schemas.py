"""Add lineage DAG tables and schema_versions for persistent storage.

Revision ID: 008
Revises: 007
Create Date: 2026-03-15

Creates:
- lineage_nodes: DAG nodes (data sources, features, models, predictions, trades)
- lineage_edges: Directed edges between nodes with unique constraint
- lineage_audit_log: Append-only audit trail for lineage operations
- schema_versions: Schema registry entries (survives Railway ephemeral FS)
"""
from alembic import op
import sqlalchemy as sa

revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- Lineage DAG: nodes ---
    op.create_table(
        "lineage_nodes",
        sa.Column("id", sa.String(200), primary_key=True),
        sa.Column("node_type", sa.String(50), nullable=False, index=True),
        sa.Column("name", sa.String(500), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("version", sa.String(50), nullable=False, server_default="1"),
        sa.Column("metadata", sa.Text(), server_default="{}"),
    )

    # --- Lineage DAG: edges ---
    op.create_table(
        "lineage_edges",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("source_id", sa.String(200), nullable=False, index=True),
        sa.Column("target_id", sa.String(200), nullable=False, index=True),
        sa.Column("edge_type", sa.String(50), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("metadata", sa.Text(), server_default="{}"),
        sa.UniqueConstraint("source_id", "target_id", "edge_type", name="uq_lineage_edge"),
    )

    # --- Lineage audit log ---
    op.create_table(
        "lineage_audit_log",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("action", sa.String(100), nullable=False, index=True),
        sa.Column("node_id", sa.String(200), nullable=False, index=True),
        sa.Column("timestamp", sa.String(50), nullable=False),
        sa.Column("details", sa.Text(), server_default="{}"),
    )

    # --- Schema versions (replaces ephemeral filesystem storage) ---
    op.create_table(
        "schema_versions",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(200), unique=True, nullable=False, index=True),
        sa.Column("version", sa.BigInteger(), nullable=False, server_default="1"),
        sa.Column("schema", sa.Text(), nullable=False),
        sa.Column("hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.String(50), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("schema_versions")
    op.drop_table("lineage_audit_log")
    op.drop_table("lineage_edges")
    op.drop_table("lineage_nodes")
