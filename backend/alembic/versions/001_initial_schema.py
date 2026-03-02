"""Initial schema - all tables for TradeMaster.

Revision ID: 001_initial
Revises:
Create Date: 2026-02-28

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- OHLCV (market data) ---
    op.create_table(
        "ohlcv",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("interval", sa.String(5), nullable=False),
        sa.Column("open_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Numeric(20, 8), nullable=False),
        sa.Column("high", sa.Numeric(20, 8), nullable=False),
        sa.Column("low", sa.Numeric(20, 8), nullable=False),
        sa.Column("close", sa.Numeric(20, 8), nullable=False),
        sa.Column("volume", sa.Numeric(20, 8), nullable=False),
        sa.Column("close_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("quote_volume", sa.Numeric(20, 8), server_default="0"),
        sa.Column("trade_count", sa.Integer(), server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ohlcv_symbol_interval_time", "ohlcv", ["symbol", "interval", "open_time"])
    op.create_index("ix_ohlcv_open_time", "ohlcv", ["open_time"])
    # Partial index for recent data queries
    op.create_index(
        "ix_ohlcv_symbol_interval_time_unique",
        "ohlcv",
        ["symbol", "interval", "open_time"],
        unique=True,
    )

    # --- Orders ---
    op.create_table(
        "orders",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("exchange_order_id", sa.String(64)),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("side", sa.String(10), nullable=False),
        sa.Column("order_type", sa.String(30), nullable=False),
        sa.Column("status", sa.String(20), server_default="PENDING"),
        sa.Column("quantity", sa.Numeric(20, 8), nullable=False),
        sa.Column("price", sa.Numeric(20, 8)),
        sa.Column("stop_price", sa.Numeric(20, 8)),
        sa.Column("filled_quantity", sa.Numeric(20, 8), server_default="0"),
        sa.Column("avg_fill_price", sa.Numeric(20, 8)),
        sa.Column("commission", sa.Numeric(20, 8), server_default="0"),
        sa.Column("signal_id", sa.BigInteger()),
        sa.Column("notes", sa.String(500)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_orders_symbol_status", "orders", ["symbol", "status"])
    op.create_index("ix_orders_exchange_id", "orders", ["exchange_order_id"])
    # Partial index for pending/open orders
    op.create_index(
        "ix_orders_pending",
        "orders",
        ["symbol", "status"],
        postgresql_where=sa.text("status IN ('PENDING', 'SUBMITTED')"),
    )

    # --- Trades ---
    op.create_table(
        "trades",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("order_id", sa.BigInteger(), nullable=False),
        sa.Column("exchange_trade_id", sa.String(64)),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("side", sa.String(10), nullable=False),
        sa.Column("price", sa.Numeric(20, 8), nullable=False),
        sa.Column("quantity", sa.Numeric(20, 8), nullable=False),
        sa.Column("commission", sa.Numeric(20, 8), server_default="0"),
        sa.Column("commission_asset", sa.String(10), server_default="'USDT'"),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_trades_symbol_time", "trades", ["symbol", "executed_at"])

    # --- Positions ---
    op.create_table(
        "positions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("side", sa.String(10), nullable=False),
        sa.Column("entry_price", sa.Numeric(20, 8), nullable=False),
        sa.Column("quantity", sa.Numeric(20, 8), nullable=False),
        sa.Column("current_price", sa.Numeric(20, 8), server_default="0"),
        sa.Column("unrealized_pnl", sa.Numeric(20, 8), server_default="0"),
        sa.Column("realized_pnl", sa.Numeric(20, 8), server_default="0"),
        sa.Column("stop_loss_price", sa.Numeric(20, 8)),
        sa.Column("take_profit_price", sa.Numeric(20, 8)),
        sa.Column("is_open", sa.Boolean(), server_default="true"),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_positions_symbol_open", "positions", ["symbol", "is_open"])
    # Partial index for open positions only
    op.create_index(
        "ix_positions_open_only",
        "positions",
        ["symbol"],
        postgresql_where=sa.text("is_open = true"),
    )

    # --- Portfolio Snapshots ---
    op.create_table(
        "portfolio_snapshots",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("total_equity", sa.Numeric(20, 8), nullable=False),
        sa.Column("available_balance", sa.Numeric(20, 8), nullable=False),
        sa.Column("unrealized_pnl", sa.Numeric(20, 8), server_default="0"),
        sa.Column("realized_pnl_cumulative", sa.Numeric(20, 8), server_default="0"),
        sa.Column("open_positions_count", sa.Integer(), server_default="0"),
        sa.Column("drawdown", sa.Numeric(10, 6), server_default="0"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_snapshots_time", "portfolio_snapshots", ["timestamp"])

    # --- Prediction Signals ---
    op.create_table(
        "prediction_signals",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("action", sa.String(10), nullable=False),
        sa.Column("strength", sa.Numeric(5, 4), nullable=False),
        sa.Column("confidence", sa.Numeric(5, 4), nullable=False),
        sa.Column("model_source", sa.String(50), nullable=False),
        sa.Column("timeframe", sa.String(5), nullable=False),
        sa.Column("features_snapshot", sa.Text()),
        sa.Column("was_executed", sa.Boolean(), server_default="false"),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_signals_symbol_time", "prediction_signals", ["symbol", "generated_at"])
    op.create_index("ix_signals_action", "prediction_signals", ["action"])

    # --- Model Metadata ---
    op.create_table(
        "model_metadata",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("model_type", sa.String(50), nullable=False),
        sa.Column("version", sa.String(20), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("accuracy", sa.Numeric(6, 4)),
        sa.Column("sharpe_ratio", sa.Numeric(8, 4)),
        sa.Column("profit_factor", sa.Numeric(8, 4)),
        sa.Column("artifact_path", sa.String(500), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default="false"),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("training_samples", sa.Integer()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("model_metadata")
    op.drop_table("prediction_signals")
    op.drop_table("portfolio_snapshots")
    op.drop_table("positions")
    op.drop_table("trades")
    op.drop_table("orders")
    op.drop_table("ohlcv")
