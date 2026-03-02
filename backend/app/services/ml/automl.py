"""AutoML model selection for automated trading model optimization."""

import time
from dataclasses import dataclass

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelEvaluation:
    """Result of evaluating a single model type."""
    model_type: str
    accuracy: float
    f1_score: float
    sharpe_proxy: float
    training_time_s: float
    params_count: int


class AutoModelSelector:
    """
    Automated model selection for trading prediction.

    Evaluates multiple model types using cross-validation and
    selects the best performer for each symbol.
    """

    AVAILABLE_MODELS = ["lstm", "xgboost", "ensemble"]

    def __init__(self):
        self._evaluation_history: list[dict] = []
        logger.info("automl_selector_initialized")

    async def select_best_model(self, symbol: str, X: np.ndarray,
                                 y: np.ndarray, n_folds: int = 5) -> dict:
        """
        Evaluate all available model types and select the best one.

        Args:
            symbol: Trading symbol
            X: Feature matrix
            y: Target labels
            n_folds: Cross-validation folds

        Returns:
            Dict with best model type and evaluation results
        """
        results: list[ModelEvaluation] = []

        for model_type in self.AVAILABLE_MODELS:
            try:
                evaluation = await self._evaluate_model(model_type, X, y, n_folds)
                results.append(evaluation)
                logger.info("model_evaluated", symbol=symbol, model=model_type,
                           accuracy=round(evaluation.accuracy, 4))
            except Exception as e:
                logger.warning("model_evaluation_failed", model=model_type, error=str(e))

        if not results:
            return {"error": "All model evaluations failed"}

        # Score: weighted combination of accuracy, F1, and Sharpe proxy
        for r in results:
            r.sharpe_proxy = r.accuracy * 0.5 + r.f1_score * 0.3 + (1 - r.training_time_s / 60) * 0.2

        best = max(results, key=lambda r: r.sharpe_proxy)

        evaluation_record = {
            "symbol": symbol,
            "best_model": best.model_type,
            "best_score": round(best.sharpe_proxy, 4),
            "all_results": [
                {
                    "model": r.model_type,
                    "accuracy": round(r.accuracy, 4),
                    "f1_score": round(r.f1_score, 4),
                    "composite_score": round(r.sharpe_proxy, 4),
                    "training_time_s": round(r.training_time_s, 2),
                }
                for r in sorted(results, key=lambda r: r.sharpe_proxy, reverse=True)
            ],
            "n_folds": n_folds,
            "n_samples": len(X),
        }

        self._evaluation_history.append(evaluation_record)
        return evaluation_record

    async def _evaluate_model(self, model_type: str, X: np.ndarray,
                               y: np.ndarray, n_folds: int) -> ModelEvaluation:
        """Evaluate a single model type using cross-validation."""
        start = time.time()
        fold_accuracies = []
        fold_f1s = []

        n_samples = len(X)
        fold_size = n_samples // n_folds

        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size

            X_val = X[val_start:val_end]
            y_val = y[val_start:val_end]
            X_train = np.concatenate([X[:val_start], X[val_end:]])
            y_train = np.concatenate([y[:val_start], y[val_end:]])

            accuracy, f1 = self._train_and_evaluate(model_type, X_train, y_train, X_val, y_val)
            fold_accuracies.append(accuracy)
            fold_f1s.append(f1)

        training_time = time.time() - start

        return ModelEvaluation(
            model_type=model_type,
            accuracy=float(np.mean(fold_accuracies)),
            f1_score=float(np.mean(fold_f1s)),
            sharpe_proxy=0.0,
            training_time_s=training_time,
            params_count=self._estimate_params(model_type, X.shape[1]),
        )

    def _train_and_evaluate(self, model_type: str, X_train: np.ndarray,
                            y_train: np.ndarray, X_val: np.ndarray,
                            y_val: np.ndarray) -> tuple[float, float]:
        """Train a model and return (accuracy, f1_score)."""
        if model_type == "xgboost":
            # Simplified gradient boosting proxy
            predictions = self._simple_tree_predict(X_train, y_train, X_val)
        elif model_type == "lstm":
            # Simplified RNN proxy
            predictions = self._simple_rnn_predict(X_train, y_train, X_val)
        elif model_type == "ensemble":
            # Average of tree and RNN
            p1 = self._simple_tree_predict(X_train, y_train, X_val)
            p2 = self._simple_rnn_predict(X_train, y_train, X_val)
            predictions = ((p1 + p2) / 2 > 0.5).astype(int)
        else:
            predictions = np.zeros(len(X_val))

        accuracy = float(np.mean(predictions == y_val))

        # F1 score calculation
        tp = float(np.sum((predictions == 1) & (y_val == 1)))
        fp = float(np.sum((predictions == 1) & (y_val == 0)))
        fn = float(np.sum((predictions == 0) & (y_val == 1)))
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        return accuracy, f1

    def _simple_tree_predict(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray) -> np.ndarray:
        """Simple nearest-neighbor proxy for tree-based model."""
        predictions = np.zeros(len(X_val))
        for i, x in enumerate(X_val):
            distances = np.sum((X_train - x) ** 2, axis=1)
            k_nearest = np.argsort(distances)[:5]
            predictions[i] = np.round(np.mean(y_train[k_nearest]))
        return predictions

    def _simple_rnn_predict(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray) -> np.ndarray:
        """Simple linear proxy for RNN model."""
        # Least squares fit
        X_aug = np.column_stack([X_train, np.ones(len(X_train))])
        try:
            weights = np.linalg.lstsq(X_aug, y_train, rcond=None)[0]
            X_val_aug = np.column_stack([X_val, np.ones(len(X_val))])
            raw = X_val_aug @ weights
            return (raw > 0.5).astype(int)
        except Exception:
            return np.zeros(len(X_val))

    def _estimate_params(self, model_type: str, n_features: int) -> int:
        """Estimate parameter count for a model type."""
        estimates = {
            "xgboost": n_features * 100 * 50,  # ~trees * leaves * features
            "lstm": 4 * (n_features * 128 + 128 * 128 + 128),  # LSTM cell
            "ensemble": n_features * 100 * 50 + 4 * (n_features * 128 + 128 * 128),
        }
        return estimates.get(model_type, 0)

    def get_history(self) -> list[dict]:
        """Get evaluation history."""
        return self._evaluation_history


auto_model_selector = AutoModelSelector()
