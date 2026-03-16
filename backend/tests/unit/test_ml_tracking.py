"""Unit tests for ML experiment tracking."""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.ml.tracking import MLTracker
from app.core.metrics import MetricsRegistry


class TestMLTracker:
    """Tests for MLTracker methods using mocked database sessions."""

    @pytest.fixture
    def tracker(self):
        return MLTracker()

    @pytest.fixture
    def mock_db(self):
        """Create a mock async database session."""
        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()
        db.execute = AsyncMock()
        db.commit = AsyncMock()
        return db

    @pytest.mark.asyncio
    async def test_log_training_run(self, tracker, mock_db):
        """Verify training run is persisted with correct fields."""
        metrics_dict = {"accuracy": 0.85, "f1": 0.82, "sharpe": 1.5}
        hyperparams = {"learning_rate": 0.01, "n_estimators": 100}
        dataset_info = {"hash": "abc123", "size": 5000}

        # Mock flush to set the id attribute
        async def mock_flush():
            for call in mock_db.add.call_args_list:
                obj = call[0][0]
                if hasattr(obj, "id"):
                    obj.id = 1

        mock_db.flush = AsyncMock(side_effect=mock_flush)

        run_id = await tracker.log_training_run(
            mock_db,
            model_type="xgboost",
            symbol="BTCUSDT",
            metrics_dict=metrics_dict,
            hyperparams=hyperparams,
            dataset_info=dataset_info,
            duration_seconds=45.2,
        )

        # Verify add was called
        mock_db.add.assert_called_once()
        added_obj = mock_db.add.call_args[0][0]
        assert added_obj.model_type == "xgboost"
        assert added_obj.symbol == "BTCUSDT"
        assert json.loads(added_obj.metrics) == metrics_dict
        assert json.loads(added_obj.hyperparams) == hyperparams
        assert added_obj.dataset_hash == "abc123"
        assert added_obj.dataset_size == 5000
        assert added_obj.duration_seconds == 45.2
        assert added_obj.status == "completed"

    @pytest.mark.asyncio
    async def test_log_training_run_failed(self, tracker, mock_db):
        """Verify failed training runs are logged with error message."""
        async def mock_flush():
            for call in mock_db.add.call_args_list:
                obj = call[0][0]
                if hasattr(obj, "id"):
                    obj.id = 2

        mock_db.flush = AsyncMock(side_effect=mock_flush)

        await tracker.log_training_run(
            mock_db,
            model_type="lstm",
            symbol="ETHUSDT",
            metrics_dict={},
            hyperparams={"lr": 0.001},
            status="failed",
            error_message="OOM on GPU",
        )

        added_obj = mock_db.add.call_args[0][0]
        assert added_obj.status == "failed"
        assert added_obj.error_message == "OOM on GPU"

    @pytest.mark.asyncio
    async def test_log_prediction(self, tracker, mock_db):
        """Verify prediction is logged with all fields."""
        async def mock_flush():
            for call in mock_db.add.call_args_list:
                obj = call[0][0]
                if hasattr(obj, "id"):
                    obj.id = 42

        mock_db.flush = AsyncMock(side_effect=mock_flush)

        pred_id = await tracker.log_prediction(
            mock_db,
            model_type="ensemble",
            symbol="BTCUSDT",
            signal="BUY",
            confidence=0.87,
            features_hash="feat123",
            latency_ms=12.5,
        )

        mock_db.add.assert_called_once()
        added_obj = mock_db.add.call_args[0][0]
        assert added_obj.model_type == "ensemble"
        assert added_obj.symbol == "BTCUSDT"
        assert added_obj.signal == "BUY"
        assert added_obj.confidence == 0.87
        assert added_obj.features_hash == "feat123"
        assert added_obj.latency_ms == 12.5

    @pytest.mark.asyncio
    async def test_log_prediction_lowercase_normalized(self, tracker, mock_db):
        """Verify symbol and signal are uppercased."""
        async def mock_flush():
            for call in mock_db.add.call_args_list:
                obj = call[0][0]
                if hasattr(obj, "id"):
                    obj.id = 43

        mock_db.flush = AsyncMock(side_effect=mock_flush)

        await tracker.log_prediction(
            mock_db,
            model_type="xgboost",
            symbol="btcusdt",
            signal="buy",
            confidence=0.5,
        )

        added_obj = mock_db.add.call_args[0][0]
        assert added_obj.symbol == "BTCUSDT"
        assert added_obj.signal == "BUY"

    @pytest.mark.asyncio
    async def test_update_prediction_outcome_success(self, tracker, mock_db):
        """Verify outcome update returns True when prediction exists."""
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_db.execute = AsyncMock(return_value=mock_result)

        updated = await tracker.update_prediction_outcome(
            mock_db, prediction_id=42, actual_outcome="BUY", outcome_pnl=150.0
        )
        assert updated is True

    @pytest.mark.asyncio
    async def test_update_prediction_outcome_not_found(self, tracker, mock_db):
        """Verify outcome update returns False when prediction doesn't exist."""
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_db.execute = AsyncMock(return_value=mock_result)

        updated = await tracker.update_prediction_outcome(
            mock_db, prediction_id=9999, actual_outcome="SELL"
        )
        assert updated is False

    @pytest.mark.asyncio
    async def test_get_rolling_accuracy_no_data(self, tracker, mock_db):
        """Verify rolling accuracy returns 0.0 when no predictions exist."""
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_db.execute = AsyncMock(return_value=mock_result)

        accuracy = await tracker.get_rolling_accuracy(
            mock_db, model_type="ensemble", symbol="BTCUSDT"
        )
        assert accuracy == 0.0

    @pytest.mark.asyncio
    async def test_get_rolling_accuracy_with_data(self, tracker, mock_db):
        """Verify rolling accuracy is calculated correctly."""
        # 7 correct out of 10 = 0.7 accuracy
        mock_rows = [
            ("BUY", "BUY"),    # correct
            ("SELL", "SELL"),   # correct
            ("BUY", "SELL"),   # wrong
            ("HOLD", "HOLD"),  # correct
            ("BUY", "BUY"),    # correct
            ("SELL", "BUY"),   # wrong
            ("BUY", "BUY"),    # correct
            ("HOLD", "SELL"),  # wrong
            ("SELL", "SELL"),  # correct
            ("BUY", "BUY"),    # correct
        ]
        mock_result = MagicMock()
        mock_result.all.return_value = mock_rows
        mock_db.execute = AsyncMock(return_value=mock_result)

        accuracy = await tracker.get_rolling_accuracy(
            mock_db, model_type="ensemble", symbol="BTCUSDT", window=10
        )
        assert accuracy == 0.7

    @pytest.mark.asyncio
    async def test_get_rolling_accuracy_perfect(self, tracker, mock_db):
        """Verify 100% accuracy case."""
        mock_rows = [("BUY", "BUY"), ("SELL", "SELL"), ("HOLD", "HOLD")]
        mock_result = MagicMock()
        mock_result.all.return_value = mock_rows
        mock_db.execute = AsyncMock(return_value=mock_result)

        accuracy = await tracker.get_rolling_accuracy(
            mock_db, model_type="xgboost", symbol="BTCUSDT"
        )
        assert accuracy == 1.0

    def test_hash_features_numpy(self, tracker):
        """Verify feature hashing works with numpy arrays."""
        import numpy as np

        features = np.array([1.0, 2.0, 3.0])
        h = MLTracker.hash_features(features)
        assert isinstance(h, str)
        assert len(h) == 16

        # Same input should produce same hash
        h2 = MLTracker.hash_features(np.array([1.0, 2.0, 3.0]))
        assert h == h2

        # Different input should produce different hash
        h3 = MLTracker.hash_features(np.array([4.0, 5.0, 6.0]))
        assert h != h3

    def test_hash_features_list(self, tracker):
        """Verify feature hashing works with plain lists."""
        features = [1.0, 2.0, 3.0]
        h = MLTracker.hash_features(features)
        assert isinstance(h, str)
        assert len(h) == 16

    def test_hash_features_different_inputs_differ(self, tracker):
        """Verify different inputs produce different hashes."""
        h1 = MLTracker.hash_features({"a": 1})
        h2 = MLTracker.hash_features({"a": 2})
        assert h1 != h2


class TestMetricsMLTracking:
    """Tests for the new ML tracking metrics in MetricsRegistry."""

    def test_ml_predictions_total_counter(self):
        registry = MetricsRegistry()
        registry.ml_predictions_total.inc(labels={"model_type": "xgboost", "signal": "BUY"})
        registry.ml_predictions_total.inc(labels={"model_type": "xgboost", "signal": "BUY"})
        registry.ml_predictions_total.inc(labels={"model_type": "lstm", "signal": "SELL"})

        output = registry.ml_predictions_total.to_prometheus()
        assert "ml_predictions_total" in output
        assert "xgboost" in output
        assert "lstm" in output

    def test_ml_prediction_latency_histogram(self):
        registry = MetricsRegistry()
        registry.ml_prediction_latency.observe(5.0)
        registry.ml_prediction_latency.observe(50.0)
        registry.ml_prediction_latency.observe(500.0)

        output = registry.ml_prediction_latency.to_prometheus()
        assert "ml_prediction_latency_ms" in output
        assert "_count 3" in output

    def test_ml_model_accuracy_rolling_gauge(self):
        registry = MetricsRegistry()
        registry.ml_model_accuracy_rolling.set(
            0.85, labels={"model_type": "ensemble", "symbol": "BTCUSDT"}
        )
        registry.ml_model_accuracy_rolling.set(
            0.72, labels={"model_type": "xgboost", "symbol": "ETHUSDT"}
        )

        output = registry.ml_model_accuracy_rolling.to_prometheus()
        assert "ml_model_accuracy_rolling" in output
        assert "0.85" in output
        assert "0.72" in output

    def test_ml_confidence_distribution_histogram(self):
        registry = MetricsRegistry()
        for v in [0.55, 0.67, 0.82, 0.91, 0.95]:
            registry.ml_confidence_distribution.observe(v)

        output = registry.ml_confidence_distribution.to_prometheus()
        assert "ml_confidence_distribution" in output
        assert "_count 5" in output

    def test_labeled_gauge_get(self):
        registry = MetricsRegistry()
        registry.ml_model_accuracy_rolling.set(
            0.88, labels={"model_type": "xgboost", "symbol": "BTCUSDT"}
        )
        val = registry.ml_model_accuracy_rolling.get(
            labels={"model_type": "xgboost", "symbol": "BTCUSDT"}
        )
        assert val == 0.88

    def test_labeled_gauge_default_zero(self):
        registry = MetricsRegistry()
        val = registry.ml_model_accuracy_rolling.get(
            labels={"model_type": "nonexistent", "symbol": "BTCUSDT"}
        )
        assert val == 0.0

    def test_collect_includes_new_metrics(self):
        registry = MetricsRegistry()
        output = registry.collect()
        assert "ml_predictions_total" in output
        assert "ml_prediction_latency_ms" in output
        assert "ml_model_accuracy_rolling" in output
        assert "ml_confidence_distribution" in output
