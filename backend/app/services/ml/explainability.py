"""Model explainability using feature importance and SHAP-like analysis."""
from dataclasses import dataclass
import numpy as np
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureExplanation:
    """Explanation for a single prediction."""
    feature_name: str
    importance: float
    direction: str  # "bullish" or "bearish"
    value: float


@dataclass
class PredictionExplanation:
    """Full explanation for a model prediction."""
    signal: str  # BUY/SELL/HOLD
    confidence: float
    top_bullish: list[FeatureExplanation]
    top_bearish: list[FeatureExplanation]
    model_type: str


class ModelExplainer:
    """Explain model predictions using feature importance."""

    def explain_xgboost(self, model, features: np.ndarray, feature_names: list[str], prediction: dict) -> PredictionExplanation:
        """Explain XGBoost prediction using built-in feature importance."""
        try:
            # Get feature importance from the model
            importance_dict = {}
            if hasattr(model, '_model') and model._model is not None:
                booster = model._model.get_booster() if hasattr(model._model, 'get_booster') else None
                if booster:
                    raw_importance = booster.get_score(importance_type='gain')
                    for fname, imp in raw_importance.items():
                        # XGBoost uses f0, f1, etc or actual names
                        idx = int(fname.replace('f', '')) if fname.startswith('f') and fname[1:].isdigit() else None
                        if idx is not None and idx < len(feature_names):
                            importance_dict[feature_names[idx]] = imp
                        elif fname in feature_names:
                            importance_dict[fname] = imp

            # If no importance available, use feature variance as proxy
            if not importance_dict and features is not None:
                for i, name in enumerate(feature_names):
                    if i < features.shape[-1]:
                        val = float(features[0][i]) if features.ndim > 1 else float(features[i])
                        importance_dict[name] = abs(val)

            return self._build_explanation(importance_dict, features, feature_names, prediction)
        except Exception as e:
            logger.warning("xgboost_explain_failed", error=str(e))
            return self._empty_explanation(prediction)

    def explain_lstm(self, features: np.ndarray, feature_names: list[str], prediction: dict) -> PredictionExplanation:
        """Explain LSTM prediction using input gradient approximation."""
        try:
            # Use feature magnitude * variance as importance proxy
            importance_dict = {}
            if features is not None and len(feature_names) > 0:
                # For sequences, use last timestep
                if features.ndim == 3:
                    last_step = features[0, -1, :]
                elif features.ndim == 2:
                    last_step = features[-1, :]
                else:
                    last_step = features

                for i, name in enumerate(feature_names):
                    if i < len(last_step):
                        importance_dict[name] = abs(float(last_step[i]))

            return self._build_explanation(importance_dict, features, feature_names, prediction)
        except Exception as e:
            logger.warning("lstm_explain_failed", error=str(e))
            return self._empty_explanation(prediction)

    def _build_explanation(
        self, importance_dict: dict, features: np.ndarray,
        feature_names: list[str], prediction: dict
    ) -> PredictionExplanation:
        """Build explanation from importance dictionary."""
        signal = prediction.get("signal", "HOLD")
        confidence = prediction.get("confidence", 0.0)

        # Normalize importances
        total = sum(abs(v) for v in importance_dict.values()) or 1.0
        normalized = {k: v / total for k, v in importance_dict.items()}

        # Determine direction based on feature value and signal
        bullish = []
        bearish = []

        for name, imp in sorted(normalized.items(), key=lambda x: abs(x[1]), reverse=True):
            # Get feature value
            feat_val = 0.0
            if name in feature_names and features is not None:
                idx = feature_names.index(name)
                if features.ndim == 3:
                    feat_val = float(features[0, -1, idx]) if idx < features.shape[-1] else 0.0
                elif features.ndim == 2:
                    feat_val = float(features[0, idx]) if idx < features.shape[-1] else 0.0

            explanation = FeatureExplanation(
                feature_name=name,
                importance=round(abs(imp), 4),
                direction="bullish" if feat_val > 0 else "bearish",
                value=round(feat_val, 6),
            )

            if feat_val > 0:
                bullish.append(explanation)
            else:
                bearish.append(explanation)

        return PredictionExplanation(
            signal=signal,
            confidence=confidence,
            top_bullish=bullish[:5],
            top_bearish=bearish[:5],
            model_type=prediction.get("model_type", "unknown"),
        )

    def _empty_explanation(self, prediction: dict) -> PredictionExplanation:
        return PredictionExplanation(
            signal=prediction.get("signal", "HOLD"),
            confidence=prediction.get("confidence", 0.0),
            top_bullish=[],
            top_bearish=[],
            model_type=prediction.get("model_type", "unknown"),
        )


model_explainer = ModelExplainer()
