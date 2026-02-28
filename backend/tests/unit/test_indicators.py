"""Tests for technical indicator engine."""

import numpy as np
import pandas as pd
import pytest

from app.services.indicators.engine import IndicatorEngine


@pytest.fixture
def sample_ohlcv():
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    open_ = close + np.random.randn(n) * 30
    volume = np.abs(np.random.randn(n) * 1000) + 100

    return pd.DataFrame({
        "open_time": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def test_compute_all_adds_columns(sample_ohlcv):
    engine = IndicatorEngine()
    result = engine.compute_all(sample_ohlcv)

    # Should have many more columns than the original
    assert len(result.columns) > len(sample_ohlcv.columns)

    # Check key indicators exist
    assert "sma_20" in result.columns
    assert "rsi_14" in result.columns
    assert "bb_upper" in result.columns
    assert "obv" in result.columns
    assert "macd" in result.columns
    assert "atr_14" in result.columns


def test_trend_indicators_present(sample_ohlcv):
    engine = IndicatorEngine()
    result = engine.compute_selective(sample_ohlcv, categories=["trend"])

    expected = ["sma_20", "sma_50", "ema_9", "ema_21", "macd", "macd_signal", "macd_hist",
                "adx", "supertrend", "hma_20"]
    for col in expected:
        assert col in result.columns, f"Missing trend indicator: {col}"


def test_momentum_indicators_present(sample_ohlcv):
    engine = IndicatorEngine()
    result = engine.compute_selective(sample_ohlcv, categories=["momentum"])

    expected = ["rsi_14", "stoch_k", "stoch_d", "cci_20", "williams_r", "roc_12",
                "mfi_14", "awesome_osc"]
    for col in expected:
        assert col in result.columns, f"Missing momentum indicator: {col}"


def test_volatility_indicators_present(sample_ohlcv):
    engine = IndicatorEngine()
    result = engine.compute_selective(sample_ohlcv, categories=["volatility"])

    expected = ["bb_upper", "bb_middle", "bb_lower", "bb_width",
                "atr_14", "atr_normalized", "keltner_upper",
                "donchian_upper", "historical_vol_20"]
    for col in expected:
        assert col in result.columns, f"Missing volatility indicator: {col}"


def test_volume_indicators_present(sample_ohlcv):
    engine = IndicatorEngine()
    result = engine.compute_selective(sample_ohlcv, categories=["volume"])

    expected = ["obv", "vwap", "cmf_20", "force_index", "volume_sma_20", "volume_ratio"]
    for col in expected:
        assert col in result.columns, f"Missing volume indicator: {col}"


def test_rsi_bounds(sample_ohlcv):
    engine = IndicatorEngine()
    result = engine.compute_all(sample_ohlcv)
    rsi = result["rsi_14"].dropna()
    assert rsi.min() >= 0, "RSI should be >= 0"
    assert rsi.max() <= 100, "RSI should be <= 100"


def test_empty_dataframe():
    engine = IndicatorEngine()
    result = engine.compute_all(pd.DataFrame())
    assert result.empty


def test_list_indicators():
    indicators = IndicatorEngine.list_indicators()
    assert "trend" in indicators
    assert "momentum" in indicators
    assert "volatility" in indicators
    assert "volume" in indicators
    assert len(indicators["trend"]) > 5
