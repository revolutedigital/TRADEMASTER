"""Market data models: OHLCV candles, trades, order book snapshots."""

from datetime import datetime

from sqlalchemy import BigInteger, Index, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class OHLCV(Base, TimestampMixin):
    """Candlestick (OHLCV) data - primary time-series table."""

    __tablename__ = "ohlcv"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    interval: Mapped[str] = mapped_column(String(5), nullable=False)  # 1m, 5m, 15m, 1h, 4h, 1d
    open_time: Mapped[datetime] = mapped_column(nullable=False)
    open: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    high: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    low: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    close: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    volume: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    close_time: Mapped[datetime] = mapped_column(nullable=False)
    quote_volume: Mapped[float] = mapped_column(Numeric(20, 8), default=0)
    trade_count: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (
        Index("ix_ohlcv_symbol_interval_time", "symbol", "interval", "open_time"),
        Index("ix_ohlcv_open_time", "open_time"),
    )

    def __repr__(self) -> str:
        return f"<OHLCV {self.symbol} {self.interval} {self.open_time}>"
