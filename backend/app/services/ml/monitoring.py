"""ML model monitoring: drift detection and performance tracking."""

from datetime import datetime, timezone
from collections import deque

import numpy as np
from scipy import stats

from app.core.logging import get_logger

logger = get_logger(__name__)


class DriftDetector:
    """Detect model prediction drift using statistical tests."""

    def __init__(self, window_size: int = 500, p_threshold: float = 0.05):
        self.window_size = window_size
        self.p_threshold = p_threshold
        self._baseline_predictions: list[float] = []
        self._recent_predictions: deque[float] = deque(maxlen=window_size)

    def set_baseline(self, predictions: list[float]):
        """Set baseline distribution from training/validation predictions."""
        self._baseline_predictions = predictions
        logger.info("drift_baseline_set", n_samples=len(predictions))

    def add_prediction(self, prediction: float):
        """Add a new prediction to the recent window."""
        self._recent_predictions.append(prediction)

    def check_drift(self) -> dict:
        """Run KS test to check for prediction distribution drift."""
        if len(self._baseline_predictions) < 50 or len(self._recent_predictions) < 50:
            return {"drift_detected": False, "reason": "insufficient_data"}

        recent = list(self._recent_predictions)
        ks_stat, p_value = stats.ks_2samp(recent, self._baseline_predictions)

        drift_detected = p_value < self.p_threshold

        if drift_detected:
            logger.warning(
                "model_drift_detected",
                ks_stat=round(ks_stat, 4),
                p_value=round(p_value, 6),
            )

        return {
            "drift_detected": drift_detected,
            "ks_statistic": round(ks_stat, 4),
            "p_value": round(p_value, 6),
            "recent_mean": round(np.mean(recent), 4),
            "baseline_mean": round(np.mean(self._baseline_predictions), 4),
            "recent_std": round(np.std(recent), 4),
            "baseline_std": round(np.std(self._baseline_predictions), 4),
            "n_recent": len(recent),
            "n_baseline": len(self._baseline_predictions),
        }


class ModelPerformanceTracker:
    """Track live model performance metrics."""

    def __init__(self):
        self._predictions: list[dict] = []

    def record_prediction(self, symbol: str, action: str, confidence: float, signal_strength: float):
        self._predictions.append({
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
            "signal_strength": signal_strength,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def get_metrics(self) -> dict:
        if not self._predictions:
            return {"total_predictions": 0}

        confidences = [p["confidence"] for p in self._predictions]
        actions = [p["action"] for p in self._predictions]

        return {
            "total_predictions": len(self._predictions),
            "avg_confidence": round(np.mean(confidences), 4),
            "buy_count": actions.count("BUY"),
            "sell_count": actions.count("SELL"),
            "hold_count": actions.count("HOLD"),
            "confidence_p25": round(np.percentile(confidences, 25), 4),
            "confidence_p75": round(np.percentile(confidences, 75), 4),
        }


drift_detector = DriftDetector()
performance_tracker = ModelPerformanceTracker()
