"""Record the exact base quantity covered by each native Spot OCO.

Revision ID: 013
Revises: 012
Create Date: 2026-08-31
"""

import sqlalchemy as sa

from alembic import op

revision = "013"
down_revision = "012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("orders", sa.Column("protective_quantity", sa.Numeric(20, 8)))
    op.add_column("positions", sa.Column("protective_quantity", sa.Numeric(20, 8)))


def downgrade() -> None:
    op.drop_column("positions", "protective_quantity")
    op.drop_column("orders", "protective_quantity")
