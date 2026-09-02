"""Retain public closed-candle taker flow for pattern research.

Revision ID: 018
Revises: 017
Create Date: 2026-09-02
"""

import sqlalchemy as sa

from alembic import op

revision = "018"
down_revision = "017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "ohlcv",
        sa.Column(
            "taker_buy_base",
            sa.Numeric(precision=20, scale=8),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )
    op.add_column(
        "ohlcv",
        sa.Column(
            "taker_buy_quote",
            sa.Numeric(precision=20, scale=8),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )


def downgrade() -> None:
    op.drop_column("ohlcv", "taker_buy_quote")
    op.drop_column("ohlcv", "taker_buy_base")
