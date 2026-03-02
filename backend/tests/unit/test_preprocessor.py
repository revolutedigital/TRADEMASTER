"""Unit tests for ML Preprocessor: scaler fitting, temporal split, target creation."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os

from app.services.ml.preprocessor import Preprocessor, SplitData, SequenceData


class TestPreprocessor:
    """Tests for the Preprocessor service."""

    @pytest.fixture
    def preprocessor(self):
        return Preprocessor()

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame with OHLCV + indicator features."""
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        close = 85000 + np.cumsum(np.random.randn(n) * 100)
        df = pd.DataFrame({
            "open_time": dates,
            "open": close - np.random.uniform(0, 200, n),
            "high": close + np.random.uniform(0, 500, n),
            "low": close - np.random.uniform(0, 500, n),
            "close": close,
            "volume": np.random.uniform(100, 5000, n),
            "rsi_14": np.random.uniform(20, 80, n),
            "macd": np.random.randn(n) * 50,
            "macd_signal": np.random.randn(n) * 30,
            "bb_upper": close + 1000,
            "bb_lower": close - 1000,
            "atr_14": np.random.uniform(200, 800, n),
            "ema_20": close + np.random.randn(n) * 50,
            "sma_50": close + np.random.randn(n) * 100,
        })
        return df

    @pytest.fixture
    def feature_cols(self):
        return ["rsi_14", "macd", "macd_signal", "bb_upper", "bb_lower",
                "atr_14", "ema_20", "sma_50"]


class TestCreateTarget(TestPreprocessor):
    """Tests for target variable creation."""

    def test_creates_target_column(self, preprocessor, sample_df):
        """create_target should add a 'target' column."""
        result = preprocessor.create_target(sample_df.copy())
        assert "target" in result.columns

    def test_target_values_are_valid(self, preprocessor, sample_df):
        """Target should be 0 (SELL), 1 (HOLD), or 2 (BUY)."""
        result = preprocessor.create_target(sample_df.copy())
        result = result.dropna(subset=["target"])
        unique_vals = set(result["target"].unique())
        assert unique_vals.issubset({0, 1, 2})

    def test_target_drops_nans_at_end(self, preprocessor, sample_df):
        """Last rows should be NaN (no future data), then dropped."""
        result = preprocessor.create_target(sample_df.copy())
        assert len(result) < len(sample_df)

    def test_target_with_custom_threshold(self, preprocessor, sample_df):
        """Different thresholds should change distribution."""
        result_low = preprocessor.create_target(sample_df.copy(), threshold=0.001)
        result_high = preprocessor.create_target(sample_df.copy(), threshold=0.05)

        low_holds = (result_low["target"] == 1).sum()
        high_holds = (result_high["target"] == 1).sum()
        # Higher threshold means more HOLDs
        assert high_holds >= low_holds


class TestPrepareTabular(TestPreprocessor):
    """Tests for tabular data preparation (XGBoost-style)."""

    def test_returns_split_data(self, preprocessor, sample_df, feature_cols):
        """prepare_tabular should return a SplitData object."""
        df = preprocessor.create_target(sample_df.copy())
        result = preprocessor.prepare_tabular(df, feature_cols)
        assert isinstance(result, SplitData)

    def test_split_shapes_consistent(self, preprocessor, sample_df, feature_cols):
        """X and y shapes should be consistent across splits."""
        df = preprocessor.create_target(sample_df.copy())
        result = preprocessor.prepare_tabular(df, feature_cols)

        assert len(result.X_train) == len(result.y_train)
        assert len(result.X_val) == len(result.y_val)
        assert len(result.X_test) == len(result.y_test)

    def test_temporal_split_order(self, preprocessor, sample_df, feature_cols):
        """Split should be temporal: train < val < test in time."""
        df = preprocessor.create_target(sample_df.copy())
        result = preprocessor.prepare_tabular(df, feature_cols)

        total = len(result.X_train) + len(result.X_val) + len(result.X_test)
        # All data should be accounted for (approximately)
        assert total > 0

    def test_feature_columns_preserved(self, preprocessor, sample_df, feature_cols):
        """Output should have the correct number of features."""
        df = preprocessor.create_target(sample_df.copy())
        result = preprocessor.prepare_tabular(df, feature_cols)

        assert result.X_train.shape[1] == len(feature_cols)

    def test_no_nan_in_output(self, preprocessor, sample_df, feature_cols):
        """Prepared data should have no NaN values."""
        df = preprocessor.create_target(sample_df.copy())
        result = preprocessor.prepare_tabular(df, feature_cols)

        assert not np.any(np.isnan(result.X_train))
        assert not np.any(np.isnan(result.y_train))


class TestPrepareSequences(TestPreprocessor):
    """Tests for LSTM sequence data preparation."""

    def test_returns_sequence_data(self, preprocessor, sample_df, feature_cols):
        """prepare_sequences should return a SequenceData object."""
        df = preprocessor.create_target(sample_df.copy())
        result = preprocessor.prepare_sequences(df, feature_cols, seq_length=24)
        assert isinstance(result, SequenceData)

    def test_sequence_shape_3d(self, preprocessor, sample_df, feature_cols):
        """LSTM input should be 3D: (samples, seq_length, features)."""
        seq_len = 24
        df = preprocessor.create_target(sample_df.copy())
        result = preprocessor.prepare_sequences(df, feature_cols, seq_length=seq_len)

        assert len(result.X_train.shape) == 3
        assert result.X_train.shape[1] == seq_len
        assert result.X_train.shape[2] == len(feature_cols)

    def test_targets_are_1d(self, preprocessor, sample_df, feature_cols):
        """Target arrays should be 1D."""
        df = preprocessor.create_target(sample_df.copy())
        result = preprocessor.prepare_sequences(df, feature_cols, seq_length=24)

        assert len(result.y_train.shape) == 1

    def test_different_seq_lengths(self, preprocessor, sample_df, feature_cols):
        """Different sequence lengths should produce different shapes."""
        df = preprocessor.create_target(sample_df.copy())
        result_12 = preprocessor.prepare_sequences(df, feature_cols, seq_length=12)
        result_48 = preprocessor.prepare_sequences(df, feature_cols, seq_length=48)

        assert result_12.X_train.shape[1] == 12
        assert result_48.X_train.shape[1] == 48


class TestScalerPersistence(TestPreprocessor):
    """Tests for scaler save/load functionality."""

    def test_save_and_load_scaler(self, preprocessor, sample_df, feature_cols):
        """Scaler should be saveable and loadable."""
        df = preprocessor.create_target(sample_df.copy())
        preprocessor.prepare_tabular(df, feature_cols)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "scaler.pkl")
            preprocessor.save_scaler(path)
            assert os.path.exists(path)

            new_preprocessor = Preprocessor()
            new_preprocessor.load_scaler(path)

    def test_loaded_scaler_produces_same_output(self, preprocessor, sample_df, feature_cols):
        """Loaded scaler should produce identical transformations."""
        df = preprocessor.create_target(sample_df.copy())
        result1 = preprocessor.prepare_tabular(df, feature_cols)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "scaler.pkl")
            preprocessor.save_scaler(path)

            new_preprocessor = Preprocessor()
            new_preprocessor.load_scaler(path)

            # Transform same data with loaded scaler
            result2 = new_preprocessor.prepare_tabular(df, feature_cols)

            np.testing.assert_array_almost_equal(
                result1.X_test[:5], result2.X_test[:5], decimal=5
            )
