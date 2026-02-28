"""ML signal and model metadata models."""

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Index, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class PredictionSignal(Base, TimestampMixin):
    __tablename__ = "prediction_signals"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    action: Mapped[str] = mapped_column(String(10), nullable=False)  # BUY, SELL, HOLD
    strength: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)  # -1.0 to +1.0
    confidence: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)  # 0.0 to 1.0
    model_source: Mapped[str] = mapped_column(String(50), nullable=False)  # ensemble, lstm, etc.
    timeframe: Mapped[str] = mapped_column(String(5), nullable=False)
    features_snapshot: Mapped[str | None] = mapped_column(Text)  # JSON of key features
    was_executed: Mapped[bool] = mapped_column(default=False)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_signals_symbol_time", "symbol", "generated_at"),
        Index("ix_signals_action", "action"),
    )


class ModelMetadata(Base, TimestampMixin):
    __tablename__ = "model_metadata"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)  # lstm, xgboost, etc.
    version: Mapped[str] = mapped_column(String(20), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    accuracy: Mapped[float | None] = mapped_column(Numeric(6, 4))
    sharpe_ratio: Mapped[float | None] = mapped_column(Numeric(8, 4))
    profit_factor: Mapped[float | None] = mapped_column(Numeric(8, 4))
    artifact_path: Mapped[str] = mapped_column(String(500), nullable=False)
    is_active: Mapped[bool] = mapped_column(default=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    training_samples: Mapped[int | None] = mapped_column()
