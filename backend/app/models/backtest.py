"""Backtest result persistence model."""

from sqlalchemy import BigInteger, Float, Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class BacktestResult(Base, TimestampMixin):
    """Persisted backtest run with configuration and results."""

    __tablename__ = "backtest_results"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    interval: Mapped[str] = mapped_column(String(5), nullable=False)
    initial_capital: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    signal_threshold: Mapped[float] = mapped_column(Numeric(4, 2), nullable=False)
    atr_stop_multiplier: Mapped[float] = mapped_column(Numeric(4, 2), nullable=False)
    risk_reward_ratio: Mapped[float] = mapped_column(Numeric(4, 2), nullable=False)
    strategy_name: Mapped[str] = mapped_column(String(100), default="ML ensemble", nullable=False)
    execution_profile: Mapped[str] = mapped_column(
        String(30), default="model_long_short", nullable=False
    )
    strategy_config_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)

    # Results
    total_return: Mapped[float] = mapped_column(Float, default=0.0)
    total_trades: Mapped[int] = mapped_column(Integer, default=0)
    winning_trades: Mapped[int] = mapped_column(Integer, default=0)
    losing_trades: Mapped[int] = mapped_column(Integer, default=0)
    win_rate: Mapped[float] = mapped_column(Float, default=0.0)
    total_return_pct: Mapped[float] = mapped_column(Float, default=0.0)
    sharpe_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    max_drawdown: Mapped[float] = mapped_column(Float, default=0.0)
    max_drawdown_pct: Mapped[float] = mapped_column(Float, default=0.0)
    profit_factor: Mapped[float] = mapped_column(Float, default=0.0)
    expectancy: Mapped[float] = mapped_column(Float, default=0.0)
    equity_curve_json: Mapped[str | None] = mapped_column(Text)  # JSON-encoded equity curve
