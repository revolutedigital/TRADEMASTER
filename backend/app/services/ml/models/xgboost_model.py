"""XGBoost model for tabular feature-based classification: BUY/HOLD/SELL."""

from pathlib import Path

import numpy as np
import xgboost as xgb

from app.core.logging import get_logger
from app.services.ml.models.base import BaseTradingModel, ModelPrediction, TrainingResult

logger = get_logger(__name__)


class XGBoostTradingModel(BaseTradingModel):
    """XGBoost classifier implementing the BaseTradingModel interface."""

    def __init__(self):
        self._model: xgb.XGBClassifier | None = None
        self._feature_names: list[str] = []

    @property
    def model_type(self) -> str:
        return "xgboost"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, features: np.ndarray) -> ModelPrediction:
        """Predict on a single sample.

        Args:
            features: shape (n_features,) or (1, n_features)
        """
        if not self._model:
            raise RuntimeError("Model not loaded. Call load() or train() first.")

        if features.ndim == 1:
            features = features.reshape(1, -1)

        probs = self._model.predict_proba(features)[0]
        action = int(np.argmax(probs))

        return ModelPrediction(
            action=action,
            probabilities=probs,
            confidence=float(probs[action]),
            signal_strength=self.probabilities_to_signal(probs),
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        feature_names: list[str] | None = None,
    ) -> TrainingResult:
        """Train XGBoost classifier."""
        # Calculate class weights
        class_counts = np.bincount(y_train, minlength=3).astype(float)
        total = class_counts.sum()
        sample_weights = np.array([total / (3 * max(c, 1)) for c in class_counts])

        self._model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            early_stopping_rounds=20,
            tree_method="hist",
            random_state=42,
            verbosity=0,
        )

        # Build sample weight array
        weight_train = np.array([sample_weights[y] for y in y_train])

        self._model.fit(
            X_train,
            y_train,
            sample_weight=weight_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        if feature_names:
            self._feature_names = feature_names

        # Metrics
        train_pred = self._model.predict(X_train)
        val_pred = self._model.predict(X_val)
        train_acc = float(np.mean(train_pred == y_train))
        val_acc = float(np.mean(val_pred == y_val))

        best_iteration = self._model.best_iteration or n_estimators

        logger.info(
            "xgboost_trained",
            train_acc=round(train_acc, 4),
            val_acc=round(val_acc, 4),
            best_iteration=best_iteration,
            n_features=X_train.shape[1],
        )

        return TrainingResult(
            accuracy=train_acc,
            loss=0.0,
            val_accuracy=val_acc,
            val_loss=0.0,
            epochs_trained=best_iteration,
            best_epoch=best_iteration,
        )

    def get_feature_importance(self, top_n: int = 20) -> dict[str, float]:
        """Get top N most important features."""
        if not self._model:
            return {}

        importances = self._model.feature_importances_
        if self._feature_names and len(self._feature_names) == len(importances):
            pairs = sorted(
                zip(self._feature_names, importances), key=lambda x: x[1], reverse=True
            )
        else:
            pairs = sorted(
                [(f"f{i}", imp) for i, imp in enumerate(importances)],
                key=lambda x: x[1],
                reverse=True,
            )

        return dict(pairs[:top_n])

    def save(self, path: Path) -> None:
        if not self._model:
            raise RuntimeError("No model to save.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))
        logger.info("xgboost_model_saved", path=str(path))

    def load(self, path: Path) -> None:
        self._model = xgb.XGBClassifier()
        self._model.load_model(str(path))
        logger.info("xgboost_model_loaded", path=str(path))
