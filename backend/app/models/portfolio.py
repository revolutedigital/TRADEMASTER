"""Portfolio and position models."""

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Index, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class Position(Base, TimestampMixin):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)  # LONG or SHORT
    entry_price: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    quantity: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    current_price: Mapped[float] = mapped_column(Numeric(20, 8), default=0)
    unrealized_pnl: Mapped[float] = mapped_column(Numeric(20, 8), default=0)
    realized_pnl: Mapped[float] = mapped_column(Numeric(20, 8), default=0)
    stop_loss_price: Mapped[float | None] = mapped_column(Numeric(20, 8))
    take_profit_price: Mapped[float | None] = mapped_column(Numeric(20, 8))
    is_open: Mapped[bool] = mapped_column(default=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_positions_symbol_open", "symbol", "is_open"),
    )

    def __repr__(self) -> str:
        status = "OPEN" if self.is_open else "CLOSED"
        return f"<Position {self.symbol} {self.side} {status}>"


class PortfolioSnapshot(Base):
    """Periodic snapshot of portfolio state for equity curve tracking."""

    __tablename__ = "portfolio_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    total_equity: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    available_balance: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    unrealized_pnl: Mapped[float] = mapped_column(Numeric(20, 8), default=0)
    realized_pnl_cumulative: Mapped[float] = mapped_column(Numeric(20, 8), default=0)
    open_positions_count: Mapped[int] = mapped_column(default=0)
    drawdown: Mapped[float] = mapped_column(Numeric(10, 6), default=0)

    __table_args__ = (Index("ix_snapshots_time", "timestamp"),)
