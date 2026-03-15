"""Add foreign keys and CHECK constraints for data integrity.

Revision ID: 005_add_fk_and_check_constraints
Revises: 004_materialized_views
Create Date: 2026-03-15 00:00:00.000000
"""

from alembic import op

revision = "005_add_fk_and_check_constraints"
down_revision = "004_materialized_views"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # FK: trades.order_id -> orders.id
    op.create_foreign_key(
        "fk_trades_order_id",
        "trades",
        "orders",
        ["order_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # CHECK constraints on orders
    op.create_check_constraint("ck_orders_side", "orders", "side IN ('BUY', 'SELL')")
    op.create_check_constraint("ck_orders_quantity_positive", "orders", "quantity > 0")

    # CHECK constraints on trades
    op.create_check_constraint("ck_trades_side", "trades", "side IN ('BUY', 'SELL')")
    op.create_check_constraint("ck_trades_price_positive", "trades", "price > 0")
    op.create_check_constraint("ck_trades_quantity_positive", "trades", "quantity > 0")

    # CHECK constraint on ohlcv
    op.create_check_constraint("ck_ohlcv_high_gte_low", "ohlcv", "high >= low")


def downgrade() -> None:
    op.drop_constraint("ck_ohlcv_high_gte_low", "ohlcv", type_="check")
    op.drop_constraint("ck_trades_quantity_positive", "trades", type_="check")
    op.drop_constraint("ck_trades_price_positive", "trades", type_="check")
    op.drop_constraint("ck_trades_side", "trades", type_="check")
    op.drop_constraint("ck_orders_quantity_positive", "orders", type_="check")
    op.drop_constraint("ck_orders_side", "orders", type_="check")
    op.drop_constraint("fk_trades_order_id", "trades", type_="foreignkey")
