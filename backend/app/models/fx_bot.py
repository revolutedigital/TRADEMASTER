"""Forex bots and what they did: one row per bot, per trade, per equity snapshot and per event.

Nothing here holds a broker credential: `broker_account_ref` is an opaque reference to a secret
kept elsewhere. The runner's append-only journal stays the raw record; these tables are what the
platform reads to show each bot's return, time in position and good and bad trades.
"""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class FxBot(Base, TimestampMixin):
    __tablename__ = "fx_bots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(40), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    strategy_key: Mapped[str] = mapped_column(String(8), nullable=False)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    timeframe_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    params_json: Mapped[str] = mapped_column(Text, nullable=False)
    mode: Mapped[str] = mapped_column(String(6), nullable=False)
    status: Mapped[str] = mapped_column(String(10), nullable=False, default="stopped")
    risk_fraction: Mapped[Decimal] = mapped_column(Numeric(8, 6), nullable=False)
    max_daily_loss_fraction: Mapped[Decimal] = mapped_column(Numeric(8, 6), nullable=False)
    broker_account_ref: Mapped[str | None] = mapped_column(String(64))

    __table_args__ = (
        CheckConstraint("mode IN ('demo', 'live')", name="ck_fx_bots_mode"),
        CheckConstraint("status IN ('stopped', 'running', 'killed')", name="ck_fx_bots_status"),
        CheckConstraint("risk_fraction > 0 AND risk_fraction <= 1", name="ck_fx_bots_risk"),
    )


class FxTrade(Base, TimestampMixin):
    __tablename__ = "fx_trades"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    bot_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("fx_bots.id", ondelete="RESTRICT"), nullable=False
    )
    client_order_id: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    broker_position_id: Mapped[str | None] = mapped_column(String(64))
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    side: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    units: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(10), nullable=False)
    refusal_reason: Mapped[str | None] = mapped_column(Text)
    stop_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 8))
    target_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 8))
    entry_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    entry_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 8))
    exit_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    exit_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 8))
    exit_reason: Mapped[str | None] = mapped_column(String(60))
    pnl: Mapped[Decimal | None] = mapped_column(Numeric(18, 4))
    r_multiple: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    commission: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))

    __table_args__ = (
        CheckConstraint("side IN (1, -1)", name="ck_fx_trades_side"),
        CheckConstraint("status IN ('intent', 'open', 'closed', 'refused')", name="ck_fx_trades_status"),
        Index("ix_fx_trades_bot_entry", "bot_id", "entry_time"),
    )


class FxEquitySnapshot(Base):
    __tablename__ = "fx_equity_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    bot_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("fx_bots.id", ondelete="RESTRICT"), nullable=False
    )
    taken_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    balance: Mapped[Decimal] = mapped_column(Numeric(18, 4), nullable=False)
    equity: Mapped[Decimal] = mapped_column(Numeric(18, 4), nullable=False)
    day_loss: Mapped[Decimal] = mapped_column(Numeric(18, 4), nullable=False)

    __table_args__ = (Index("ix_fx_equity_bot_time", "bot_id", "taken_at"),)


class FxBotEvent(Base):
    __tablename__ = "fx_bot_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    bot_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("fx_bots.id", ondelete="RESTRICT"), nullable=False
    )
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    kind: Mapped[str] = mapped_column(String(30), nullable=False)
    detail: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    __table_args__ = (Index("ix_fx_events_bot_time", "bot_id", "occurred_at"),)
