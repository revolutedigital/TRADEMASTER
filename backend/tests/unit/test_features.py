"""Unit tests for feature engineering."""

import pytest
import numpy as np
import pandas as pd

from app.services.ml.features import FeatureEngineer


@pytest.fixture
def feature_engineer():
    return FeatureEngineer()


@pytest.fixture
def sufficient_ohlcv():
    """Create a DataFrame with enough data for feature engineering."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2025-01-01", periods=n, freq="h")
    base_price = 85000
    prices = base_price + np.cumsum(np.random.randn(n) * 100)

    df = pd.DataFrame({
        "open_time": dates,
        "open": prices,
        "high": prices + np.abs(np.random.randn(n) * 200),
        "low": prices - np.abs(np.random.randn(n) * 200),
        "close": prices + np.random.randn(n) * 50,
        "volume": np.abs(np.random.randn(n) * 100) + 10,
    })
    df["high"] = df[["open", "high", "close"]].max(axis=1) + 10
    df["low"] = df[["open", "low", "close"]].min(axis=1) - 10
    return df


class TestFeatureEngineer:
    def test_build_features_adds_columns(self, feature_engineer, sufficient_ohlcv):
        result = feature_engineer.build_features(sufficient_ohlcv)
        assert len(result.columns) > len(sufficient_ohlcv.columns)

    def test_build_features_insufficient_data(self, feature_engineer):
        small_df = pd.DataFrame({
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "close": [101, 102],
            "volume": [10, 11],
        })
        result = feature_engineer.build_features(small_df)
        # Should return as-is without adding features
        assert len(result.columns) == len(small_df.columns)

    def test_build_features_empty_df(self, feature_engineer):
        empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = feature_engineer.build_features(empty_df)
        assert result.empty

    def test_price_features_created(self, feature_engineer, sufficient_ohlcv):
        result = feature_engineer.build_features(sufficient_ohlcv)
        price_cols = [c for c in result.columns if "price_vs_" in c or "pctrank" in c]
        assert len(price_cols) > 0

    def test_return_features_created(self, feature_engineer, sufficient_ohlcv):
        result = feature_engineer.build_features(sufficient_ohlcv)
        return_cols = [c for c in result.columns if "return" in c.lower() or "ret_" in c]
        assert len(return_cols) > 0

    def test_volume_features_created(self, feature_engineer, sufficient_ohlcv):
        result = feature_engineer.build_features(sufficient_ohlcv)
        vol_cols = [c for c in result.columns if "vol" in c.lower() and c not in sufficient_ohlcv.columns]
        assert len(vol_cols) > 0

    def test_no_inf_values(self, feature_engineer, sufficient_ohlcv):
        result = feature_engineer.build_features(sufficient_ohlcv)
        numeric = result.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any(), "Feature engineering produced infinite values"
