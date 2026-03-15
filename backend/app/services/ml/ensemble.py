"""Ensemble model voting for trading signals."""
from dataclasses import dataclass

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EnsembleVote:
    """Result of ensemble voting."""
    signal: str  # BUY / SELL / HOLD
    confidence: float
    agreement: float  # 0-1 how much models agree
    model_votes: dict[str, dict]  # Individual model predictions
    weights_used: dict[str, float]


class EnsembleVoter:
    """Weighted voting across multiple model predictions."""

    # Default weights per model type
    DEFAULT_WEIGHTS = {
        "xgboost": 0.45,
        "lstm": 0.35,
        "simple_signal": 0.20,
    }

    SIGNAL_MAP = {"BUY": 1, "HOLD": 0, "SELL": -1}
    REVERSE_MAP = {1: "BUY", 0: "HOLD", -1: "SELL"}

    def __init__(self, weights: dict[str, float] | None = None):
        self._weights = weights or self.DEFAULT_WEIGHTS.copy()

    def vote(self, predictions: list[dict]) -> EnsembleVote:
        """Combine multiple model predictions into a single signal.

        Each prediction dict should have:
            - model_type: str (xgboost, lstm, simple_signal)
            - signal: str (BUY, SELL, HOLD)
            - confidence: float (0-1)
        """
        if not predictions:
            return EnsembleVote(
                signal="HOLD", confidence=0.0, agreement=0.0,
                model_votes={}, weights_used=self._weights,
            )

        # Calculate weighted score
        total_weight = 0.0
        weighted_score = 0.0
        weighted_confidence = 0.0
        model_votes = {}

        for pred in predictions:
            model_type = pred.get("model_type", "unknown")
            signal = pred.get("signal", "HOLD")
            confidence = pred.get("confidence", 0.5)
            weight = self._weights.get(model_type, 0.1)

            signal_val = self.SIGNAL_MAP.get(signal, 0)
            weighted_score += signal_val * confidence * weight
            weighted_confidence += confidence * weight
            total_weight += weight

            model_votes[model_type] = {
                "signal": signal,
                "confidence": round(confidence, 4),
                "weight": weight,
            }

        if total_weight == 0:
            return EnsembleVote(
                signal="HOLD", confidence=0.0, agreement=0.0,
                model_votes=model_votes, weights_used=self._weights,
            )

        # Normalize
        avg_score = weighted_score / total_weight
        avg_confidence = weighted_confidence / total_weight

        # Determine final signal based on score thresholds
        if avg_score > 0.15:
            final_signal = "BUY"
        elif avg_score < -0.15:
            final_signal = "SELL"
        else:
            final_signal = "HOLD"

        # Calculate agreement (how much models agree on direction)
        signals = [self.SIGNAL_MAP.get(p.get("signal", "HOLD"), 0) for p in predictions]
        if len(signals) > 1:
            # Agreement = 1 if all same, 0 if split
            unique_signals = set(signals)
            if len(unique_signals) == 1:
                agreement = 1.0
            elif len(unique_signals) == 2:
                # Partial agreement
                most_common = max(set(signals), key=signals.count)
                agreement = signals.count(most_common) / len(signals)
            else:
                agreement = 0.0
        else:
            agreement = 1.0

        # Conservative confidence: reduce when models disagree
        final_confidence = avg_confidence * agreement

        return EnsembleVote(
            signal=final_signal,
            confidence=round(final_confidence, 4),
            agreement=round(agreement, 4),
            model_votes=model_votes,
            weights_used=self._weights,
        )

    def update_weights(self, model_type: str, performance_score: float) -> None:
        """Adjust weight for a model based on recent performance."""
        if model_type in self._weights:
            # Exponential moving average of performance
            current = self._weights[model_type]
            alpha = 0.1  # Learning rate
            self._weights[model_type] = current * (1 - alpha) + performance_score * alpha

            # Renormalize
            total = sum(self._weights.values())
            if total > 0:
                self._weights = {k: v / total for k, v in self._weights.items()}

            logger.info("ensemble_weight_updated", model=model_type,
                       new_weight=round(self._weights[model_type], 4))


ensemble_voter = EnsembleVoter()
