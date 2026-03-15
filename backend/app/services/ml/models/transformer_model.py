"""Transformer-based trading model with temporal attention."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    pass


@dataclass
class TransformerResult:
    predictions: list[float]  # Multi-horizon predictions
    attention_weights: list[float]  # Which timesteps matter most
    confidence: float
    horizons: list[int]  # e.g. [1, 5, 10] candles ahead


class NumpyAttention:
    """Lightweight self-attention using only numpy (no torch needed)."""

    def __init__(self, d_model: int = 64, n_heads: int = 4):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        # Random init weights (would be loaded from trained model)
        rng = np.random.RandomState(42)
        self.W_q = rng.randn(d_model, d_model).astype(np.float32) * 0.02
        self.W_k = rng.randn(d_model, d_model).astype(np.float32) * 0.02
        self.W_v = rng.randn(d_model, d_model).astype(np.float32) * 0.02
        self.W_o = rng.randn(d_model, d_model).astype(np.float32) * 0.02

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """x: (seq_len, d_model) -> output: (seq_len, d_model), attn: (seq_len, seq_len)"""
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # Scaled dot-product attention
        scores = (Q @ K.T) / np.sqrt(self.d_k)
        # Softmax
        exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

        output = attn_weights @ V
        output = output @ self.W_o
        return output, attn_weights


class TemporalFusionPredictor:
    """Temporal Fusion Transformer-inspired predictor.

    Uses self-attention to weight historical timesteps and produce
    multi-horizon predictions (1, 5, 10 candles ahead).
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, horizons: list[int] | None = None):
        self.d_model = d_model
        self.horizons = horizons or [1, 5, 10]
        self.attention = NumpyAttention(d_model=d_model, n_heads=n_heads)
        self._is_trained = False
        # Output projection (d_model -> len(horizons))
        rng = np.random.RandomState(42)
        self.output_proj = rng.randn(d_model, len(self.horizons)).astype(np.float32) * 0.02

    def predict(self, features: np.ndarray) -> TransformerResult:
        """Generate multi-horizon predictions.

        Args:
            features: (seq_len, n_features) array of historical features
        """
        seq_len, n_features = features.shape

        # Project features to d_model dimension
        if n_features != self.d_model:
            # Simple linear projection
            rng = np.random.RandomState(hash(n_features) % 2**31)
            proj = rng.randn(n_features, self.d_model).astype(np.float32) * 0.02
            x = features.astype(np.float32) @ proj
        else:
            x = features.astype(np.float32)

        # Apply self-attention
        attended, attn_weights = self.attention.forward(x)

        # Use last timestep for prediction
        last_hidden = attended[-1]  # (d_model,)

        # Multi-horizon output
        raw_pred = last_hidden @ self.output_proj  # (n_horizons,)

        # Sigmoid for probability-like output
        predictions = 1.0 / (1.0 + np.exp(-raw_pred))

        # Temporal attention weights (how much each timestep contributed)
        temporal_importance = attn_weights[-1].tolist()  # Last query's attention over all keys

        # Confidence based on attention concentration (high entropy = low confidence)
        attn_entropy = -np.sum(attn_weights[-1] * np.log(attn_weights[-1] + 1e-10))
        max_entropy = np.log(seq_len)
        confidence = 1.0 - (attn_entropy / max_entropy) if max_entropy > 0 else 0.5

        return TransformerResult(
            predictions=predictions.tolist(),
            attention_weights=temporal_importance,
            confidence=float(np.clip(confidence, 0, 1)),
            horizons=self.horizons,
        )

    def save(self, path: Path) -> None:
        """Save model weights."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "d_model": self.d_model,
            "horizons": self.horizons,
            "W_q": self.attention.W_q.tolist(),
            "W_k": self.attention.W_k.tolist(),
            "W_v": self.attention.W_v.tolist(),
            "W_o": self.attention.W_o.tolist(),
            "output_proj": self.output_proj.tolist(),
            "is_trained": self._is_trained,
        }
        path.write_text(json.dumps(data))
        logger.info("transformer_saved", path=str(path))

    def load(self, path: Path) -> bool:
        """Load model weights."""
        path = Path(path)
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text())
            self.d_model = data["d_model"]
            self.horizons = data["horizons"]
            self.attention.W_q = np.array(data["W_q"], dtype=np.float32)
            self.attention.W_k = np.array(data["W_k"], dtype=np.float32)
            self.attention.W_v = np.array(data["W_v"], dtype=np.float32)
            self.attention.W_o = np.array(data["W_o"], dtype=np.float32)
            self.output_proj = np.array(data["output_proj"], dtype=np.float32)
            self._is_trained = data.get("is_trained", False)
            logger.info("transformer_loaded", path=str(path))
            return True
        except Exception as e:
            logger.warning("transformer_load_failed", error=str(e))
            return False


# Singleton
transformer_predictor = TemporalFusionPredictor()
