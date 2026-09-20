"""Forex bots, their trades, equity snapshots and events (additive; nothing existing is touched).

Revision ID: 019
Revises: 018
Create Date: 2026-09-20

Rollback: `alembic downgrade 018` drops only the four tables created here.
"""

import sqlalchemy as sa

from alembic import op

revision = "019"
down_revision = "018"
branch_labels = None
depends_on = None

_now = sa.text("CURRENT_TIMESTAMP")


def _timestamps() -> list[sa.Column]:
    return [
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=_now),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=_now),
    ]


def upgrade() -> None:
    op.create_table(
        "fx_bots",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), primary_key=True, autoincrement=True),
        sa.Column("key", sa.String(40), nullable=False, unique=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("strategy_key", sa.String(8), nullable=False),
        sa.Column("symbol", sa.String(10), nullable=False),
        sa.Column("timeframe_seconds", sa.Integer(), nullable=False),
        sa.Column("params_json", sa.Text(), nullable=False),
        sa.Column("mode", sa.String(6), nullable=False),
        sa.Column("status", sa.String(10), nullable=False, server_default="stopped"),
        sa.Column("risk_fraction", sa.Numeric(8, 6), nullable=False),
        sa.Column("max_daily_loss_fraction", sa.Numeric(8, 6), nullable=False),
        sa.Column("broker_account_ref", sa.String(64)),
        *_timestamps(),
        sa.CheckConstraint("mode IN ('demo', 'live')", name="ck_fx_bots_mode"),
        sa.CheckConstraint("status IN ('stopped', 'running', 'killed')", name="ck_fx_bots_status"),
        sa.CheckConstraint("risk_fraction > 0 AND risk_fraction <= 1", name="ck_fx_bots_risk"),
    )
    bot_id = lambda: sa.Column(  # noqa: E731
        "bot_id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
        sa.ForeignKey("fx_bots.id", ondelete="RESTRICT"), nullable=False,
    )
    identity = lambda: sa.Column(  # noqa: E731
        "id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), primary_key=True, autoincrement=True
    )
    op.create_table(
        "fx_trades",
        identity(),
        bot_id(),
        sa.Column("client_order_id", sa.String(80), nullable=False, unique=True),
        sa.Column("broker_position_id", sa.String(64)),
        sa.Column("symbol", sa.String(10), nullable=False),
        sa.Column("side", sa.SmallInteger(), nullable=False),
        sa.Column("units", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(10), nullable=False),
        sa.Column("refusal_reason", sa.Text()),
        sa.Column("stop_price", sa.Numeric(18, 8)),
        sa.Column("target_price", sa.Numeric(18, 8)),
        sa.Column("entry_time", sa.DateTime(timezone=True)),
        sa.Column("entry_price", sa.Numeric(18, 8)),
        sa.Column("exit_time", sa.DateTime(timezone=True)),
        sa.Column("exit_price", sa.Numeric(18, 8)),
        sa.Column("exit_reason", sa.String(60)),
        sa.Column("pnl", sa.Numeric(18, 4)),
        sa.Column("r_multiple", sa.Numeric(10, 4)),
        sa.Column("commission", sa.Numeric(12, 4)),
        *_timestamps(),
        sa.CheckConstraint("side IN (1, -1)", name="ck_fx_trades_side"),
        sa.CheckConstraint("status IN ('intent', 'open', 'closed', 'refused')", name="ck_fx_trades_status"),
    )
    op.create_index("ix_fx_trades_bot_entry", "fx_trades", ["bot_id", "entry_time"])
    op.create_table(
        "fx_equity_snapshots",
        identity(),
        bot_id(),
        sa.Column("taken_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("balance", sa.Numeric(18, 4), nullable=False),
        sa.Column("equity", sa.Numeric(18, 4), nullable=False),
        sa.Column("day_loss", sa.Numeric(18, 4), nullable=False),
    )
    op.create_index("ix_fx_equity_bot_time", "fx_equity_snapshots", ["bot_id", "taken_at"])
    op.create_table(
        "fx_bot_events",
        identity(),
        bot_id(),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("kind", sa.String(30), nullable=False),
        sa.Column("detail", sa.Text(), nullable=False, server_default="{}"),
    )
    op.create_index("ix_fx_events_bot_time", "fx_bot_events", ["bot_id", "occurred_at"])


def downgrade() -> None:
    op.drop_index("ix_fx_events_bot_time", table_name="fx_bot_events")
    op.drop_table("fx_bot_events")
    op.drop_index("ix_fx_equity_bot_time", table_name="fx_equity_snapshots")
    op.drop_table("fx_equity_snapshots")
    op.drop_index("ix_fx_trades_bot_entry", table_name="fx_trades")
    op.drop_table("fx_trades")
    op.drop_table("fx_bots")
