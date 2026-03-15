"""Confidence calibration for ML model predictions."""
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CalibrationResult:
    """Result of confidence calibration."""
    raw_confidence: float
    calibrated_confidence: float
    reliability: float  # How reliable is this calibration (0-1)


class ConfidenceCalibrator:
    """Platt-scaling inspired confidence calibration.

    Maps raw model confidence to calibrated probabilities based on
    historical prediction accuracy at different confidence levels.
    """

    def __init__(self):
        # Bin edges for confidence buckets
        self._bin_edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # Track predictions and outcomes per bin
        self._bin_correct: dict[int, int] = {}
        self._bin_total: dict[int, int] = {}
        # Calibration map: raw_bin -> calibrated_confidence
        self._calibration_map: dict[int, float] = {}
        self._min_samples = 20  # Minimum samples per bin to trust calibration

    def _get_bin(self, confidence: float) -> int:
        """Get bin index for a confidence value."""
        for i in range(len(self._bin_edges) - 1):
            if confidence <= self._bin_edges[i + 1]:
                return i
        return len(self._bin_edges) - 2

    def record_outcome(self, raw_confidence: float, was_correct: bool) -> None:
        """Record a prediction outcome for calibration learning."""
        bin_idx = self._get_bin(raw_confidence)
        self._bin_total[bin_idx] = self._bin_total.get(bin_idx, 0) + 1
        if was_correct:
            self._bin_correct[bin_idx] = self._bin_correct.get(bin_idx, 0) + 1

        # Update calibration map
        total = self._bin_total[bin_idx]
        if total >= self._min_samples:
            correct = self._bin_correct.get(bin_idx, 0)
            self._calibration_map[bin_idx] = correct / total

    def calibrate(self, raw_confidence: float) -> CalibrationResult:
        """Calibrate a raw confidence value."""
        bin_idx = self._get_bin(raw_confidence)
        total = self._bin_total.get(bin_idx, 0)

        if bin_idx in self._calibration_map:
            calibrated = self._calibration_map[bin_idx]
            reliability = min(1.0, total / (self._min_samples * 5))
        else:
            # Not enough data: apply conservative shrinkage toward 0.33 (random for 3 classes)
            base_rate = 1.0 / 3.0
            shrinkage = max(0.0, 1.0 - total / self._min_samples)
            calibrated = raw_confidence * (1 - shrinkage) + base_rate * shrinkage
            reliability = 0.0 if total == 0 else min(0.5, total / self._min_samples)

        return CalibrationResult(
            raw_confidence=round(raw_confidence, 4),
            calibrated_confidence=round(calibrated, 4),
            reliability=round(reliability, 4),
        )

    def get_calibration_stats(self) -> dict:
        """Get calibration statistics for monitoring."""
        stats = {}
        for i in range(len(self._bin_edges) - 1):
            bin_label = f"{self._bin_edges[i]:.1f}-{self._bin_edges[i+1]:.1f}"
            total = self._bin_total.get(i, 0)
            correct = self._bin_correct.get(i, 0)
            stats[bin_label] = {
                "total_predictions": total,
                "correct_predictions": correct,
                "observed_accuracy": round(correct / total, 4) if total > 0 else None,
                "calibrated_confidence": self._calibration_map.get(i),
                "is_calibrated": i in self._calibration_map,
            }
        return stats

    def save(self, path: Path) -> None:
        """Save calibration state to disk."""
        data = {
            "bin_correct": {str(k): v for k, v in self._bin_correct.items()},
            "bin_total": {str(k): v for k, v in self._bin_total.items()},
            "calibration_map": {str(k): v for k, v in self._calibration_map.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
        logger.info("calibration_saved", path=str(path))

    def load(self, path: Path) -> bool:
        """Load calibration state from disk."""
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text())
            self._bin_correct = {int(k): v for k, v in data.get("bin_correct", {}).items()}
            self._bin_total = {int(k): v for k, v in data.get("bin_total", {}).items()}
            self._calibration_map = {int(k): v for k, v in data.get("calibration_map", {}).items()}
            logger.info("calibration_loaded", path=str(path), bins=len(self._calibration_map))
            return True
        except Exception as e:
            logger.warning("calibration_load_failed", error=str(e))
            return False


# Per-symbol calibrators
_calibrators: dict[str, ConfidenceCalibrator] = {}


def get_calibrator(symbol: str) -> ConfidenceCalibrator:
    """Get or create calibrator for a symbol."""
    if symbol not in _calibrators:
        _calibrators[symbol] = ConfidenceCalibrator()
        # Try to load from disk
        path = Path(f"ml_artifacts/calibration/{symbol.lower()}_calibration.json")
        _calibrators[symbol].load(path)
    return _calibrators[symbol]
