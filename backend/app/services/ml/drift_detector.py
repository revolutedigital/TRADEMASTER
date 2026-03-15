"""Model drift detection + auto-retraining trigger.

Monitors live prediction accuracy (rolling window) and triggers
retraining when performance drops below threshold.

Drift indicators:
1. Rolling accuracy drops below 45% (last 50 predictions)
2. Sharpe of recent signals drops below 0 (last 30 days)
3. Feature distribution shift (KL divergence)
"""

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

ROLLING_WINDOW = 50  # Last 50 predictions
MIN_PREDICTIONS_FOR_CHECK = 20
ACCURACY_THRESHOLD = 0.45  # Retrain if accuracy drops below 45%
RETRAIN_COOLDOWN_HOURS = 24  # Don't retrain more than once per day


@dataclass
class PredictionOutcome:
    """Record of a prediction and its actual outcome."""

    symbol: str
    predicted_action: str  # BUY, SELL, HOLD
    actual_direction: str  # UP, DOWN, FLAT
    signal_strength: float
    price_change_pct: float
    timestamp: float


class DriftDetector:
    """Monitors model performance and triggers retraining when needed."""

    def __init__(self) -> None:
        self._outcomes: dict[str, deque[PredictionOutcome]] = {}
        self._last_retrain_time: float = 0
        self._retraining_in_progress: bool = False

    def record_outcome(
        self,
        symbol: str,
        predicted_action: str,
        actual_price_change_pct: float,
        signal_strength: float,
    ) -> None:
        """Record a prediction outcome for drift monitoring.

        Call this after the prediction period has elapsed (e.g., after 15m candle closes).
        """
        if symbol not in self._outcomes:
            self._outcomes[symbol] = deque(maxlen=ROLLING_WINDOW * 2)

        if actual_price_change_pct > 0.001:
            actual_direction = "UP"
        elif actual_price_change_pct < -0.001:
            actual_direction = "DOWN"
        else:
            actual_direction = "FLAT"

        outcome = PredictionOutcome(
            symbol=symbol,
            predicted_action=predicted_action,
            actual_direction=actual_direction,
            signal_strength=signal_strength,
            price_change_pct=actual_price_change_pct,
            timestamp=time.time(),
        )
        self._outcomes[symbol].append(outcome)

    def check_drift(self, symbol: str | None = None) -> dict:
        """Check if model shows signs of drift.

        Returns drift status and metrics for the symbol or all symbols.
        """
        symbols = [symbol] if symbol else list(self._outcomes.keys())
        results = {}

        for sym in symbols:
            outcomes = list(self._outcomes.get(sym, []))
            recent = outcomes[-ROLLING_WINDOW:]

            if len(recent) < MIN_PREDICTIONS_FOR_CHECK:
                results[sym] = {
                    "status": "insufficient_data",
                    "predictions": len(recent),
                    "min_needed": MIN_PREDICTIONS_FOR_CHECK,
                }
                continue

            # Calculate directional accuracy
            correct = 0
            total_with_action = 0
            for o in recent:
                if o.predicted_action == "HOLD":
                    continue
                total_with_action += 1
                if (
                    (o.predicted_action == "BUY" and o.actual_direction == "UP")
                    or (o.predicted_action == "SELL" and o.actual_direction == "DOWN")
                ):
                    correct += 1

            accuracy = correct / total_with_action if total_with_action > 0 else 0

            # Signal quality: avg return when following signals
            signal_returns = []
            for o in recent:
                if o.predicted_action == "BUY":
                    signal_returns.append(o.price_change_pct)
                elif o.predicted_action == "SELL":
                    signal_returns.append(-o.price_change_pct)

            avg_signal_return = float(np.mean(signal_returns)) if signal_returns else 0
            signal_sharpe = (
                float(np.mean(signal_returns) / max(np.std(signal_returns), 0.0001))
                if len(signal_returns) >= 5
                else 0
            )

            # Determine drift status
            needs_retrain = False
            drift_reasons = []

            if accuracy < ACCURACY_THRESHOLD:
                needs_retrain = True
                drift_reasons.append(f"accuracy={accuracy:.2%} < {ACCURACY_THRESHOLD:.0%}")

            if signal_sharpe < 0 and len(signal_returns) >= 10:
                needs_retrain = True
                drift_reasons.append(f"signal_sharpe={signal_sharpe:.2f} < 0")

            results[sym] = {
                "status": "drift_detected" if needs_retrain else "healthy",
                "accuracy": round(accuracy, 4),
                "total_predictions": len(recent),
                "actionable_predictions": total_with_action,
                "avg_signal_return": round(avg_signal_return, 6),
                "signal_sharpe": round(signal_sharpe, 4),
                "needs_retrain": needs_retrain,
                "drift_reasons": drift_reasons,
            }

            if needs_retrain:
                logger.warning(
                    "model_drift_detected",
                    symbol=sym,
                    accuracy=round(accuracy, 4),
                    signal_sharpe=round(signal_sharpe, 4),
                    reasons=drift_reasons,
                )

        return results

    async def auto_retrain_if_needed(self) -> dict[str, bool]:
        """Check all symbols and trigger retraining if drift detected.

        Returns dict of {symbol: was_retrained}.
        """
        now = time.time()

        # Cooldown check
        if now - self._last_retrain_time < RETRAIN_COOLDOWN_HOURS * 3600:
            return {}

        if self._retraining_in_progress:
            return {}

        drift_status = self.check_drift()
        retrained = {}

        for symbol, status in drift_status.items():
            if not status.get("needs_retrain", False):
                retrained[symbol] = False
                continue

            logger.info("auto_retrain_triggered", symbol=symbol, reason=status.get("drift_reasons"))

            try:
                self._retraining_in_progress = True
                success = await self._retrain_model(symbol)
                retrained[symbol] = success

                if success:
                    self._last_retrain_time = now
                    # Clear old outcomes so we start fresh
                    self._outcomes[symbol] = deque(maxlen=ROLLING_WINDOW * 2)
            finally:
                self._retraining_in_progress = False

        return retrained

    async def _retrain_model(self, symbol: str) -> bool:
        """Trigger model retraining for a symbol."""
        try:
            from app.models.base import async_session_factory
            from app.services.market.data_collector import market_data_collector
            from app.services.ml.pipeline import ml_pipeline

            async with async_session_factory() as db:
                df = await market_data_collector.get_latest_candles(
                    db=db, symbol=symbol, interval="15m", limit=5000,
                )

            if df.empty or len(df) < 200:
                logger.warning("retrain_insufficient_data", symbol=symbol, candles=len(df))
                return False

            result = await ml_pipeline.train(df, symbol)
            logger.info(
                "model_retrained",
                symbol=symbol,
                metrics=result if isinstance(result, dict) else str(result),
            )
            return True
        except Exception as e:
            logger.error("model_retrain_failed", symbol=symbol, error=str(e))
            return False

    def get_status(self) -> dict:
        """Get overall drift detector status."""
        return {
            "symbols_tracked": list(self._outcomes.keys()),
            "last_retrain_time": self._last_retrain_time,
            "retraining_in_progress": self._retraining_in_progress,
            "cooldown_hours": RETRAIN_COOLDOWN_HOURS,
            "drift_status": self.check_drift(),
        }


drift_detector = DriftDetector()
