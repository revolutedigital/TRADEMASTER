"""Abstract base interface for all ML trading models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ModelPrediction:
    """Output of a model prediction."""

    action: int  # 0=SELL, 1=HOLD, 2=BUY
    probabilities: np.ndarray  # shape (3,) — [sell_prob, hold_prob, buy_prob]
    confidence: float  # max probability value
    signal_strength: float  # mapped to [-1.0, +1.0]

    @property
    def action_label(self) -> str:
        return {0: "SELL", 1: "HOLD", 2: "BUY"}[self.action]


@dataclass
class TrainingResult:
    """Output of model training."""

    accuracy: float
    loss: float
    val_accuracy: float
    val_loss: float
    epochs_trained: int
    best_epoch: int


class BaseTradingModel(ABC):
    """All ML models implement this interface for uniform orchestration."""

    @abstractmethod
    def predict(self, features: np.ndarray) -> ModelPrediction:
        """Return prediction with confidence score.

        Args:
            features: For tabular models, shape (n_features,) or (1, n_features).
                      For sequence models, shape (seq_len, n_features).
        """

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TrainingResult:
        """Train model, return metrics."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialize model to disk."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model from disk."""

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Identifier string (e.g., 'lstm', 'xgboost')."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model is ready for prediction."""

    @staticmethod
    def probabilities_to_signal(probs: np.ndarray) -> float:
        """Convert class probabilities to signal strength [-1.0, +1.0].

        Formula: signal = buy_prob - sell_prob
        """
        return float(probs[2] - probs[0])
