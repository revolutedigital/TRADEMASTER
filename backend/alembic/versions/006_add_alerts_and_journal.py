"""Add price_alerts and trade_journal tables.

Revision ID: 006
Revises: 005_add_fk_and_check_constraints
Create Date: 2026-03-15
"""
from alembic import op
import sqlalchemy as sa

revision = "006"
down_revision = "005_add_fk_and_check_constraints"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "price_alerts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(20), nullable=False, index=True),
        sa.Column("condition", sa.String(10), nullable=False),
        sa.Column("target_price", sa.Float(), nullable=False),
        sa.Column("is_triggered", sa.Boolean(), default=False, nullable=False),
        sa.Column("triggered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), default=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    op.create_table(
        "trade_journal",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("trade_id", sa.Integer(), sa.ForeignKey("trades.id", ondelete="SET NULL"), nullable=True),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("tags", sa.JSON(), default=list),
        sa.Column("sentiment", sa.String(20), default="neutral"),
        sa.Column("lessons_learned", sa.Text(), nullable=True),
        sa.Column("rating", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("trade_journal")
    op.drop_table("price_alerts")
