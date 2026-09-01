"""Persistent, evidence-backed technical strategy deployments."""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class StrategyDeployment(Base, TimestampMixin):
    """One reproducible strategy candidate or activation for the trading engine."""

    __tablename__ = "strategy_deployments"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source_backtest_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("backtest_results.id", ondelete="RESTRICT"),
        nullable=False,
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    interval: Mapped[str] = mapped_column(String(5), nullable=False)
    strategy_config_json: Mapped[str] = mapped_column(Text, nullable=False)
    execution_config_json: Mapped[str] = mapped_column(Text, nullable=False)
    target_execution_mode: Mapped[str] = mapped_column(String(10), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    total_test_trades: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    walk_forward_windows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_return_pct: Mapped[float] = mapped_column(nullable=False, default=0)
    avg_sharpe: Mapped[float] = mapped_column(nullable=False, default=0)
    avg_max_drawdown_pct: Mapped[float] = mapped_column(nullable=False, default=0)
    avg_profit_factor: Mapped[float] = mapped_column(nullable=False, default=0)
    consistency_score: Mapped[float] = mapped_column(nullable=False, default=0)
    overfitting_score: Mapped[float] = mapped_column(nullable=False, default=0)
    rejection_reason: Mapped[str | None] = mapped_column(Text)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deactivated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index(
            "ix_strategy_deployments_runtime",
            "symbol",
            "interval",
            "target_execution_mode",
            "status",
        ),
        Index(
            "uq_strategy_deployments_active_runtime",
            "symbol",
            "interval",
            "target_execution_mode",
            unique=True,
            postgresql_where=text("status = 'ACTIVE'"),
        ),
        CheckConstraint(
            "target_execution_mode IN ('PAPER', 'TESTNET', 'LIVE')",
            name="ck_strategy_deployments_execution_mode",
        ),
        CheckConstraint(
            "status IN ('APPROVED', 'ACTIVE', 'REJECTED', 'DISABLED')",
            name="ck_strategy_deployments_status",
        ),
    )
