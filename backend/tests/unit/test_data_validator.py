"""Tests for OHLCV data validation."""

import pytest
from app.services.market.data_validator import DataValidator


class TestDataValidator:
    def setup_method(self):
        self.validator = DataValidator()

    def test_valid_candle(self):
        result = self.validator.validate_ohlcv({
            "open": 85000, "high": 85500, "low": 84500,
            "close": 85200, "volume": 100,
        })
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_high_less_than_low(self):
        result = self.validator.validate_ohlcv({
            "open": 85000, "high": 84000, "low": 86000,
            "close": 85200, "volume": 100,
        })
        assert result.is_valid is False
        assert any("high" in e and "low" in e for e in result.errors)

    def test_negative_close(self):
        result = self.validator.validate_ohlcv({
            "open": 85000, "high": 85500, "low": -100,
            "close": -50, "volume": 100,
        })
        assert result.is_valid is False

    def test_negative_volume(self):
        result = self.validator.validate_ohlcv({
            "open": 85000, "high": 85500, "low": 84500,
            "close": 85200, "volume": -10,
        })
        assert result.is_valid is False

    def test_zero_volume_warning(self):
        result = self.validator.validate_ohlcv({
            "open": 85000, "high": 85500, "low": 84500,
            "close": 85200, "volume": 0,
        })
        assert result.is_valid is True
        assert any("zero volume" in w for w in result.warnings)

    def test_price_continuity_normal(self):
        result = self.validator.validate_price_continuity(85100, 85000)
        assert result.is_valid is True
        assert len(result.warnings) == 0

    def test_price_continuity_large_gap(self):
        result = self.validator.validate_price_continuity(100000, 85000)
        assert len(result.warnings) > 0

    def test_volume_spike_normal(self):
        result = self.validator.validate_volume_spike(150, 100)
        assert len(result.warnings) == 0

    def test_volume_spike_detected(self):
        result = self.validator.validate_volume_spike(6000, 100)
        assert len(result.warnings) > 0
