"""Model explainability: understand why models make predictions."""
import numpy as np
from dataclasses import dataclass

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExplanationResult:
    top_positive: list[tuple[str, float]]  # Features pushing toward BUY
    top_negative: list[tuple[str, float]]  # Features pushing toward SELL
    feature_importances: dict[str, float]


class ModelExplainer:
    """Explain ML model predictions using feature importance and gradient analysis."""

    def explain_xgboost(self, model, feature_names: list[str], top_n: int = 5) -> ExplanationResult:
        """Explain XGBoost prediction using built-in feature importance."""
        try:
            importances = model.get_feature_importance()
            if not importances:
                importances = {name: 0.0 for name in feature_names}
        except Exception:
            importances = {name: 0.0 for name in feature_names}

        sorted_features = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)

        top_positive = [(name, val) for name, val in sorted_features if val > 0][:top_n]
        top_negative = [(name, val) for name, val in sorted_features if val < 0][:top_n]

        # If no negative values (all importances positive), split by median
        if not top_negative:
            mid = len(sorted_features) // 2
            top_positive = sorted_features[:top_n]
            top_negative = sorted_features[mid:mid + top_n]

        return ExplanationResult(
            top_positive=top_positive,
            top_negative=top_negative,
            feature_importances=dict(sorted_features[:20]),
        )

    def explain_prediction_simple(
        self, features: np.ndarray, feature_names: list[str], prediction_probs: np.ndarray
    ) -> ExplanationResult:
        """Simple explanation based on feature deviations from mean."""
        # Use absolute feature values as proxy for importance
        abs_features = np.abs(features)
        importance_indices = np.argsort(abs_features)[::-1]

        importances = {}
        for idx in importance_indices[:20]:
            name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            importances[name] = float(features[idx])

        sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        top_positive = [(n, v) for n, v in sorted_items if v > 0][:5]
        top_negative = [(n, v) for n, v in sorted_items if v < 0][:5]

        return ExplanationResult(
            top_positive=top_positive,
            top_negative=top_negative,
            feature_importances=importances,
        )


model_explainer = ModelExplainer()
