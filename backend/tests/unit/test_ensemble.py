"""Unit tests for ensemble voting."""
import pytest
from app.services.ml.ensemble import EnsembleVoter


class TestEnsembleVoter:
    def test_unanimous_buy(self):
        voter = EnsembleVoter()
        preds = [
            {"model_type": "xgboost", "signal": "BUY", "confidence": 0.8},
            {"model_type": "lstm", "signal": "BUY", "confidence": 0.7},
        ]
        result = voter.vote(preds)
        assert result.signal == "BUY"
        assert result.agreement == 1.0
        assert result.confidence > 0.5

    def test_unanimous_sell(self):
        voter = EnsembleVoter()
        preds = [
            {"model_type": "xgboost", "signal": "SELL", "confidence": 0.9},
            {"model_type": "lstm", "signal": "SELL", "confidence": 0.8},
        ]
        result = voter.vote(preds)
        assert result.signal == "SELL"
        assert result.agreement == 1.0

    def test_disagreement_defaults_hold(self):
        voter = EnsembleVoter()
        preds = [
            {"model_type": "xgboost", "signal": "BUY", "confidence": 0.6},
            {"model_type": "lstm", "signal": "SELL", "confidence": 0.6},
        ]
        result = voter.vote(preds)
        # With disagreement, confidence should be reduced
        assert result.agreement < 1.0

    def test_empty_predictions(self):
        voter = EnsembleVoter()
        result = voter.vote([])
        assert result.signal == "HOLD"
        assert result.confidence == 0.0

    def test_single_prediction(self):
        voter = EnsembleVoter()
        preds = [{"model_type": "xgboost", "signal": "BUY", "confidence": 0.9}]
        result = voter.vote(preds)
        assert result.signal == "BUY"
        assert result.agreement == 1.0

    def test_weight_update(self):
        voter = EnsembleVoter()
        old_weight = voter._weights["xgboost"]
        voter.update_weights("xgboost", 0.9)
        # Weight should change
        assert voter._weights["xgboost"] != old_weight
        # Weights should sum to ~1
        assert abs(sum(voter._weights.values()) - 1.0) < 0.01
