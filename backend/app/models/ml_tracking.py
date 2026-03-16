"""SQLAlchemy models for ML experiment tracking and prediction logging."""

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    func,
)

from app.models.base import Base


class TrainingRun(Base):
    """Records each ML model training run with full metadata."""

    __tablename__ = "ml_training_runs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    model_type = Column(String(50), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    metrics = Column(Text, nullable=False, server_default="{}")  # JSON: accuracy, f1, sharpe, etc.
    hyperparams = Column(Text, nullable=False, server_default="{}")  # JSON: learning_rate, etc.
    dataset_hash = Column(String(64), nullable=True)
    dataset_size = Column(Integer, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    status = Column(String(20), nullable=False, server_default="completed")  # completed, failed
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class PredictionLog(Base):
    """Logs every ML prediction for accuracy tracking and analysis."""

    __tablename__ = "ml_prediction_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    model_type = Column(String(50), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    signal = Column(String(10), nullable=False)  # BUY, SELL, HOLD
    confidence = Column(Float, nullable=False)
    features_hash = Column(String(64), nullable=True)
    actual_outcome = Column(String(10), nullable=True)  # BUY, SELL, HOLD or None (pending)
    outcome_pnl = Column(Float, nullable=True)  # Realized P&L if tracked
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
