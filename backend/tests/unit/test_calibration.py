"""Unit tests for confidence calibration."""
import pytest
from app.services.ml.calibration import ConfidenceCalibrator


class TestConfidenceCalibrator:
    def test_no_data_shrinks_to_base_rate(self):
        cal = ConfidenceCalibrator()
        result = cal.calibrate(0.8)
        # With no data, should shrink toward 1/3
        assert result.calibrated_confidence < 0.8
        assert result.reliability == 0.0

    def test_calibration_after_data(self):
        cal = ConfidenceCalibrator()
        # Record 25 predictions at 0.8 confidence, 60% correct
        for i in range(25):
            cal.record_outcome(0.8, i < 15)  # 15/25 = 60%

        result = cal.calibrate(0.8)
        assert abs(result.calibrated_confidence - 0.6) < 0.05
        assert result.reliability > 0

    def test_perfect_calibration(self):
        cal = ConfidenceCalibrator()
        # All predictions at 0.7 are correct
        for _ in range(25):
            cal.record_outcome(0.7, True)

        result = cal.calibrate(0.7)
        assert result.calibrated_confidence > 0.9

    def test_stats(self):
        cal = ConfidenceCalibrator()
        for _ in range(10):
            cal.record_outcome(0.5, True)

        stats = cal.get_calibration_stats()
        assert "0.4-0.5" in stats or "0.5-0.6" in stats
