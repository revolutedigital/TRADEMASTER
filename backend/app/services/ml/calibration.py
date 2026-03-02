"""Model confidence calibration using Platt scaling."""
import numpy as np
from dataclasses import dataclass

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CalibrationResult:
    raw_confidence: float
    calibrated_confidence: float
    reliability_score: float  # How well-calibrated the model is


class ConfidenceCalibrator:
    """Calibrate model prediction probabilities using isotonic regression / Platt scaling.

    Ensures that when the model says 70% confidence, outcomes are correct ~70% of the time.
    """

    def __init__(self):
        self._calibration_map: dict[int, float] = {}  # bucket -> observed accuracy
        self._n_predictions: int = 0
        self._n_correct: int = 0
        self._history: list[tuple[float, bool]] = []  # (confidence, was_correct)

    def record(self, confidence: float, was_correct: bool) -> None:
        """Record a prediction outcome for calibration."""
        self._history.append((confidence, was_correct))
        self._n_predictions += 1
        if was_correct:
            self._n_correct += 1

        # Keep rolling window of 1000
        if len(self._history) > 1000:
            self._history = self._history[-1000:]

        # Rebuild calibration map every 100 predictions
        if self._n_predictions % 100 == 0:
            self._rebuild_calibration()

    def calibrate(self, raw_confidence: float) -> float:
        """Apply calibration to a raw confidence score."""
        if not self._calibration_map:
            return raw_confidence

        bucket = int(raw_confidence * 10)
        bucket = max(0, min(9, bucket))

        if bucket in self._calibration_map:
            return self._calibration_map[bucket]
        return raw_confidence

    def get_result(self, raw_confidence: float) -> CalibrationResult:
        calibrated = self.calibrate(raw_confidence)
        reliability = self._calculate_reliability()
        return CalibrationResult(
            raw_confidence=raw_confidence,
            calibrated_confidence=round(calibrated, 4),
            reliability_score=round(reliability, 4),
        )

    def _rebuild_calibration(self) -> None:
        """Rebuild calibration map from history."""
        buckets: dict[int, list[bool]] = {}
        for conf, correct in self._history:
            b = min(9, int(conf * 10))
            buckets.setdefault(b, []).append(correct)

        self._calibration_map = {}
        for b, outcomes in buckets.items():
            if len(outcomes) >= 5:
                self._calibration_map[b] = sum(outcomes) / len(outcomes)

    def _calculate_reliability(self) -> float:
        """Expected Calibration Error (lower is better, inverted for score)."""
        if not self._calibration_map:
            return 0.5
        errors = []
        for b, observed in self._calibration_map.items():
            expected = (b + 0.5) / 10
            errors.append(abs(observed - expected))
        ece = np.mean(errors) if errors else 0.5
        return max(0.0, 1.0 - ece)


confidence_calibrator = ConfidenceCalibrator()
