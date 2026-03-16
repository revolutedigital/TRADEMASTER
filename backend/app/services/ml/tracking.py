"""ML experiment tracker: logs training runs, predictions, and rolling accuracy.

Uses PostgreSQL for persistence (no MLflow/external deps). Designed for
production use in the TradeMaster trading pipeline.
"""

import hashlib
import json
import time
from datetime import datetime, timezone

from sqlalchemy import desc, select, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.core.metrics import metrics
from app.models.ml_tracking import PredictionLog, TrainingRun

logger = get_logger(__name__)


class MLTracker:
    """Tracks ML training runs and predictions for production monitoring."""

    async def log_training_run(
        self,
        db: AsyncSession,
        model_type: str,
        symbol: str,
        metrics_dict: dict,
        hyperparams: dict,
        dataset_info: dict | None = None,
        duration_seconds: float | None = None,
        status: str = "completed",
        error_message: str | None = None,
    ) -> int:
        """Log a completed (or failed) training run.

        Args:
            db: Async database session.
            model_type: Model identifier (xgboost, lstm, transformer, ensemble).
            symbol: Trading symbol (e.g., BTCUSDT).
            metrics_dict: Training metrics (accuracy, f1, sharpe, loss, etc.).
            hyperparams: Hyperparameters used for training.
            dataset_info: Optional dict with dataset_hash, dataset_size, etc.
            duration_seconds: Training wall-clock time.
            status: "completed" or "failed".
            error_message: Error details if status == "failed".

        Returns:
            The training run ID.
        """
        dataset_info = dataset_info or {}
        dataset_hash = dataset_info.get("hash")
        dataset_size = dataset_info.get("size")

        run = TrainingRun(
            model_type=model_type,
            symbol=symbol.upper(),
            metrics=json.dumps(metrics_dict),
            hyperparams=json.dumps(hyperparams),
            dataset_hash=dataset_hash,
            dataset_size=dataset_size,
            duration_seconds=duration_seconds,
            status=status,
            error_message=error_message,
        )
        db.add(run)
        await db.flush()

        logger.info(
            "training_run_logged",
            run_id=run.id,
            model_type=model_type,
            symbol=symbol,
            status=status,
            metrics=metrics_dict,
        )
        return run.id

    async def log_prediction(
        self,
        db: AsyncSession,
        model_type: str,
        symbol: str,
        signal: str,
        confidence: float,
        features_hash: str | None = None,
        latency_ms: float | None = None,
    ) -> int:
        """Log a single ML prediction for tracking.

        Args:
            db: Async database session.
            model_type: Model that generated the prediction.
            symbol: Trading symbol.
            signal: Predicted action (BUY, SELL, HOLD).
            confidence: Model confidence [0, 1].
            features_hash: Hash of input features for reproducibility.
            latency_ms: Inference latency in milliseconds.

        Returns:
            The prediction log ID.
        """
        pred = PredictionLog(
            model_type=model_type,
            symbol=symbol.upper(),
            signal=signal.upper(),
            confidence=confidence,
            features_hash=features_hash,
            latency_ms=latency_ms,
        )
        db.add(pred)
        await db.flush()

        # Update in-memory metrics
        metrics.ml_predictions_total.inc(
            labels={"model_type": model_type, "signal": signal.upper()}
        )
        if latency_ms is not None:
            metrics.ml_prediction_latency.observe(latency_ms)
        metrics.ml_confidence_distribution.observe(confidence)

        return pred.id

    async def update_prediction_outcome(
        self,
        db: AsyncSession,
        prediction_id: int,
        actual_outcome: str,
        outcome_pnl: float | None = None,
    ) -> bool:
        """Update a prediction with its actual outcome.

        Args:
            db: Async database session.
            prediction_id: ID of the prediction to update.
            actual_outcome: What actually happened (BUY, SELL, HOLD).
            outcome_pnl: Realized P&L if available.

        Returns:
            True if the prediction was found and updated.
        """
        values = {"actual_outcome": actual_outcome.upper()}
        if outcome_pnl is not None:
            values["outcome_pnl"] = outcome_pnl

        result = await db.execute(
            update(PredictionLog)
            .where(PredictionLog.id == prediction_id)
            .values(**values)
        )

        if result.rowcount > 0:
            logger.info(
                "prediction_outcome_updated",
                prediction_id=prediction_id,
                actual_outcome=actual_outcome,
                outcome_pnl=outcome_pnl,
            )
            return True
        return False

    async def get_rolling_accuracy(
        self,
        db: AsyncSession,
        model_type: str,
        symbol: str,
        window: int = 100,
    ) -> float:
        """Calculate rolling accuracy for the last N predictions with outcomes.

        Args:
            db: Async database session.
            model_type: Model to check.
            symbol: Trading symbol.
            window: Number of recent predictions to consider.

        Returns:
            Accuracy as a float [0, 1], or 0.0 if no data.
        """
        # Get last N predictions that have actual_outcome filled
        stmt = (
            select(PredictionLog.signal, PredictionLog.actual_outcome)
            .where(
                PredictionLog.model_type == model_type,
                PredictionLog.symbol == symbol.upper(),
                PredictionLog.actual_outcome.is_not(None),
            )
            .order_by(desc(PredictionLog.created_at))
            .limit(window)
        )

        result = await db.execute(stmt)
        rows = result.all()

        if not rows:
            return 0.0

        correct = sum(1 for signal, outcome in rows if signal == outcome)
        accuracy = correct / len(rows)

        # Update the gauge metric
        metrics.ml_model_accuracy_rolling.set(
            accuracy, labels={"model_type": model_type, "symbol": symbol.upper()}
        )

        return accuracy

    async def get_model_comparison(
        self,
        db: AsyncSession,
        symbol: str,
    ) -> list[dict]:
        """Compare all model types for a symbol by rolling accuracy and avg confidence.

        Returns a list of dicts with model_type, accuracy, avg_confidence,
        total_predictions, and latest training metrics.
        """
        symbol = symbol.upper()

        # Get distinct model types that have predictions for this symbol
        stmt = (
            select(PredictionLog.model_type)
            .where(PredictionLog.symbol == symbol)
            .distinct()
        )
        result = await db.execute(stmt)
        model_types = [row[0] for row in result.all()]

        comparisons = []
        for model_type in model_types:
            # Rolling accuracy
            accuracy = await self.get_rolling_accuracy(db, model_type, symbol)

            # Avg confidence + total predictions
            stats_stmt = select(
                func.avg(PredictionLog.confidence),
                func.count(PredictionLog.id),
                func.avg(PredictionLog.latency_ms),
            ).where(
                PredictionLog.model_type == model_type,
                PredictionLog.symbol == symbol,
            )
            stats = (await db.execute(stats_stmt)).one()
            avg_confidence = float(stats[0]) if stats[0] else 0.0
            total_predictions = int(stats[1])
            avg_latency_ms = float(stats[2]) if stats[2] else None

            # Latest training run metrics
            latest_run_stmt = (
                select(TrainingRun.metrics, TrainingRun.created_at)
                .where(
                    TrainingRun.model_type == model_type,
                    TrainingRun.symbol == symbol,
                    TrainingRun.status == "completed",
                )
                .order_by(desc(TrainingRun.created_at))
                .limit(1)
            )
            latest_run = (await db.execute(latest_run_stmt)).first()
            training_metrics = {}
            last_trained = None
            if latest_run:
                try:
                    training_metrics = json.loads(latest_run[0])
                except (json.JSONDecodeError, TypeError):
                    pass
                last_trained = latest_run[1].isoformat() if latest_run[1] else None

            comparisons.append({
                "model_type": model_type,
                "rolling_accuracy": round(accuracy, 4),
                "avg_confidence": round(avg_confidence, 4),
                "total_predictions": total_predictions,
                "avg_latency_ms": round(avg_latency_ms, 2) if avg_latency_ms else None,
                "training_metrics": training_metrics,
                "last_trained": last_trained,
            })

        # Sort by rolling accuracy descending
        comparisons.sort(key=lambda x: x["rolling_accuracy"], reverse=True)
        return comparisons

    async def get_training_runs(
        self,
        db: AsyncSession,
        model_type: str | None = None,
        symbol: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """List training runs with optional filters.

        Args:
            db: Async database session.
            model_type: Filter by model type (optional).
            symbol: Filter by symbol (optional).
            limit: Max results to return.
            offset: Pagination offset.

        Returns:
            List of training run dicts.
        """
        stmt = select(TrainingRun).order_by(desc(TrainingRun.created_at))

        if model_type:
            stmt = stmt.where(TrainingRun.model_type == model_type)
        if symbol:
            stmt = stmt.where(TrainingRun.symbol == symbol.upper())

        stmt = stmt.limit(limit).offset(offset)
        result = await db.execute(stmt)
        runs = result.scalars().all()

        return [
            {
                "id": run.id,
                "model_type": run.model_type,
                "symbol": run.symbol,
                "metrics": json.loads(run.metrics) if run.metrics else {},
                "hyperparams": json.loads(run.hyperparams) if run.hyperparams else {},
                "dataset_hash": run.dataset_hash,
                "dataset_size": run.dataset_size,
                "duration_seconds": run.duration_seconds,
                "status": run.status,
                "error_message": run.error_message,
                "created_at": run.created_at.isoformat() if run.created_at else None,
            }
            for run in runs
        ]

    @staticmethod
    def hash_features(feature_values) -> str:
        """Create a reproducible hash from feature values for tracking."""
        try:
            import numpy as np

            if hasattr(feature_values, "tobytes"):
                data = feature_values.tobytes()
            else:
                data = json.dumps(feature_values, sort_keys=True, default=str).encode()
            return hashlib.sha256(data).hexdigest()[:16]
        except Exception:
            return "unknown"


# Module-level singleton
ml_tracker = MLTracker()
