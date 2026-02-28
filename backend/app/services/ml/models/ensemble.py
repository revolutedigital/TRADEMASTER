"""Ensemble combiner: weighted average of multiple model predictions."""

import numpy as np

from app.core.logging import get_logger
from app.services.ml.models.base import BaseTradingModel, ModelPrediction

logger = get_logger(__name__)

# Signal strength thresholds
STRONG_BUY_THRESHOLD = 0.3
STRONG_SELL_THRESHOLD = -0.3


class EnsembleModel:
    """Combines predictions from multiple models using weighted average.

    Output: SignalStrength float in [-1.0, +1.0]
      [-1.0, -0.3]: SELL signal
      [-0.3, +0.3]: HOLD / no action
      [+0.3, +1.0]: BUY signal
    """

    def __init__(
        self,
        models: dict[str, BaseTradingModel],
        weights: dict[str, float] | None = None,
    ):
        self.models = models
        # Default: equal weights
        self.weights = weights or {name: 1.0 / len(models) for name in models}
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def predict(self, features: dict[str, np.ndarray]) -> ModelPrediction:
        """Combine predictions from all models.

        Args:
            features: dict mapping model_name -> feature array
                      (each model may need different input shapes)
        """
        combined_probs = np.zeros(3)
        total_weight = 0.0

        for name, model in self.models.items():
            if not model.is_loaded:
                logger.warning("model_not_loaded_skipping", model=name)
                continue

            model_features = features.get(name)
            if model_features is None:
                continue

            pred = model.predict(model_features)
            weight = self.weights.get(name, 0.0)

            # Confidence-weighted: scale weight by model's confidence
            effective_weight = weight * pred.confidence
            combined_probs += pred.probabilities * effective_weight
            total_weight += effective_weight

        # Normalize
        if total_weight > 0:
            combined_probs /= total_weight

        # Ensure valid probabilities
        combined_probs = np.clip(combined_probs, 0, 1)
        prob_sum = combined_probs.sum()
        if prob_sum > 0:
            combined_probs /= prob_sum

        action = int(np.argmax(combined_probs))
        signal = float(combined_probs[2] - combined_probs[0])

        return ModelPrediction(
            action=action,
            probabilities=combined_probs,
            confidence=float(combined_probs[action]),
            signal_strength=signal,
        )

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """Update model weights (e.g., from recent backtest performance)."""
        self.weights.update(new_weights)
        self._normalize_weights()
        logger.info("ensemble_weights_updated", weights=self.weights)

    @staticmethod
    def signal_to_action(signal: float) -> str:
        """Convert signal strength to action label."""
        if signal >= STRONG_BUY_THRESHOLD:
            return "BUY"
        elif signal <= STRONG_SELL_THRESHOLD:
            return "SELL"
        return "HOLD"
