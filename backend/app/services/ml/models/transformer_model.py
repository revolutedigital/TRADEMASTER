"""Transformer-based trading model with temporal attention and numpy training."""
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

    Training uses numpy-only backpropagation through the output projection
    layer + simple gradient updates to the attention weights.
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, horizons: list[int] | None = None):
        self.d_model = d_model
        self.horizons = horizons or [1, 5, 10]
        self.attention = NumpyAttention(d_model=d_model, n_heads=n_heads)
        self._is_trained = False
        # Output projection (d_model -> len(horizons))
        rng = np.random.RandomState(42)
        self.output_proj = rng.randn(d_model, len(self.horizons)).astype(np.float32) * 0.02
        # Input projection cache: keyed by n_features -> weight matrix
        self._input_projs: dict[int, np.ndarray] = {}
        # Training stats
        self._train_losses: list[float] = []

    # ------------------------------------------------------------------
    # Input projection (deterministic but trainable)
    # ------------------------------------------------------------------

    def _get_input_proj(self, n_features: int) -> np.ndarray:
        """Get or create a deterministic input projection matrix."""
        if n_features not in self._input_projs:
            rng = np.random.RandomState(hash(n_features) % 2**31)
            self._input_projs[n_features] = (
                rng.randn(n_features, self.d_model).astype(np.float32) * 0.02
            )
        return self._input_projs[n_features]

    # ------------------------------------------------------------------
    # Forward pass (returns intermediates for backprop during training)
    # ------------------------------------------------------------------

    def _forward(self, features: np.ndarray) -> dict:
        """Full forward pass returning all intermediates."""
        seq_len, n_features = features.shape
        x_input = features.astype(np.float32)

        # Input projection
        if n_features != self.d_model:
            input_proj = self._get_input_proj(n_features)
            x_proj = x_input @ input_proj
        else:
            input_proj = None
            x_proj = x_input

        # Attention
        attended, attn_weights = self.attention.forward(x_proj)

        # Last hidden state
        last_hidden = attended[-1]  # (d_model,)

        # Raw prediction
        raw_pred = last_hidden @ self.output_proj  # (n_horizons,)

        # Sigmoid
        predictions = 1.0 / (1.0 + np.exp(-np.clip(raw_pred, -20, 20)))

        return {
            "x_input": x_input,
            "input_proj": input_proj,
            "x_proj": x_proj,
            "attended": attended,
            "attn_weights": attn_weights,
            "last_hidden": last_hidden,
            "raw_pred": raw_pred,
            "predictions": predictions,
        }

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray) -> TransformerResult:
        """Generate multi-horizon predictions.

        Args:
            features: (seq_len, n_features) array of historical features
        """
        fwd = self._forward(features)
        predictions = fwd["predictions"]
        attn_weights = fwd["attn_weights"]

        # Temporal attention weights (how much each timestep contributed)
        temporal_importance = attn_weights[-1].tolist()

        # Confidence based on attention concentration (high entropy = low confidence)
        attn_last = attn_weights[-1]
        attn_entropy = -np.sum(attn_last * np.log(attn_last + 1e-10))
        seq_len = features.shape[0]
        max_entropy = np.log(seq_len)
        confidence = 1.0 - (attn_entropy / max_entropy) if max_entropy > 0 else 0.5

        # Trained models get a confidence boost
        if self._is_trained:
            confidence = min(1.0, confidence * 1.2)

        return TransformerResult(
            predictions=predictions.tolist(),
            attention_weights=temporal_importance,
            confidence=float(np.clip(confidence, 0, 1)),
            horizons=self.horizons,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 20,
        lr: float = 0.005,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> dict:
        """Train the model on historical data using numpy-only backpropagation.

        This implements gradient descent through the output projection and
        attention output weights via the chain rule.  It uses BCE loss for
        the sigmoid outputs.

        Args:
            X: Training features, shape (n_samples, seq_len, n_features).
               Each sample is a window of sequential candle features.
            y: Targets, shape (n_samples, n_horizons).
               Values in [0, 1] representing up/down probability at each horizon.
            epochs: Number of passes through the data.
            lr: Learning rate.
            batch_size: Mini-batch size.
            verbose: Log training progress.

        Returns:
            dict with training stats (losses per epoch).
        """
        n_samples = X.shape[0]
        n_horizons = len(self.horizons)

        if y.shape[1] != n_horizons:
            raise ValueError(
                f"Target horizons mismatch: y has {y.shape[1]} cols, "
                f"expected {n_horizons} for horizons {self.horizons}"
            )

        # Ensure input projection exists for this feature size
        n_features = X.shape[2]
        if n_features != self.d_model:
            self._get_input_proj(n_features)

        epoch_losses = []
        best_loss = float("inf")
        best_output_proj = self.output_proj.copy()
        best_W_o = self.attention.W_o.copy()

        for epoch in range(epochs):
            # Shuffle
            indices = np.random.permutation(n_samples)
            total_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start : start + batch_size]
                batch_X = X[batch_idx]
                batch_y = y[batch_idx]
                bs = len(batch_idx)

                # Accumulate gradients over batch
                grad_output_proj = np.zeros_like(self.output_proj)
                grad_W_o = np.zeros_like(self.attention.W_o)
                grad_input_proj = None
                batch_loss = 0.0

                for i in range(bs):
                    fwd = self._forward(batch_X[i])
                    pred = fwd["predictions"]  # (n_horizons,)
                    target = batch_y[i]        # (n_horizons,)

                    # BCE loss: -[t*log(p) + (1-t)*log(1-p)]
                    p = np.clip(pred, 1e-7, 1 - 1e-7)
                    loss_i = -np.mean(target * np.log(p) + (1 - target) * np.log(1 - p))
                    batch_loss += loss_i

                    # --- Backprop through sigmoid + output_proj ---
                    # d_loss/d_raw_pred = (pred - target) / n_horizons  (BCE gradient)
                    d_raw = (pred - target) / n_horizons  # (n_horizons,)

                    # d_loss/d_output_proj = last_hidden^T @ d_raw
                    last_hidden = fwd["last_hidden"]  # (d_model,)
                    grad_output_proj += np.outer(last_hidden, d_raw)

                    # d_loss/d_last_hidden = output_proj @ d_raw
                    d_last_hidden = self.output_proj @ d_raw  # (d_model,)

                    # --- Backprop through W_o ---
                    # attended output: output = (attn @ V) @ W_o
                    # last_hidden = output[-1] = pre_Wo[-1] @ W_o
                    # d_loss/d_W_o += pre_Wo[-1]^T @ d_last_hidden
                    attn_weights = fwd["attn_weights"]
                    x_proj = fwd["x_proj"]  # (seq_len, d_model)
                    V = x_proj @ self.attention.W_v  # (seq_len, d_model)
                    pre_Wo = attn_weights @ V  # (seq_len, d_model)
                    pre_Wo_last = pre_Wo[-1]  # (d_model,)
                    grad_W_o += np.outer(pre_Wo_last, d_last_hidden)

                    # --- Backprop through input projection (if used) ---
                    if fwd["input_proj"] is not None and n_features != self.d_model:
                        if grad_input_proj is None:
                            grad_input_proj = np.zeros_like(fwd["input_proj"])
                        # d_loss/d_x_proj[-1] propagates through W_o
                        # This is an approximation: we only backprop through the
                        # output path, not through the full attention recomputation
                        d_x_proj_last = d_last_hidden @ self.attention.W_o.T
                        x_input_last = fwd["x_input"][-1]  # (n_features,)
                        grad_input_proj += np.outer(x_input_last, d_x_proj_last)

                batch_loss /= bs
                total_loss += batch_loss
                n_batches += 1

                # Apply gradients (SGD)
                self.output_proj -= lr * (grad_output_proj / bs)
                self.attention.W_o -= lr * (grad_W_o / bs)
                if grad_input_proj is not None and n_features != self.d_model:
                    self._input_projs[n_features] -= lr * (grad_input_proj / bs)

            avg_loss = total_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)

            # Track best weights
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_output_proj = self.output_proj.copy()
                best_W_o = self.attention.W_o.copy()

            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                logger.info(
                    "transformer_train_epoch",
                    epoch=epoch + 1,
                    total_epochs=epochs,
                    loss=round(avg_loss, 6),
                )

        # Restore best weights
        self.output_proj = best_output_proj
        self.attention.W_o = best_W_o

        self._is_trained = True
        self._train_losses = epoch_losses

        improvement = (
            (epoch_losses[0] - best_loss) / epoch_losses[0] * 100
            if epoch_losses[0] > 0 else 0
        )
        logger.info(
            "transformer_training_complete",
            final_loss=round(best_loss, 6),
            initial_loss=round(epoch_losses[0], 6),
            improvement_pct=round(improvement, 2),
            epochs=epochs,
        )

        return {
            "epoch_losses": epoch_losses,
            "best_loss": best_loss,
            "improvement_pct": improvement,
            "is_trained": True,
        }

    @staticmethod
    def prepare_training_data(
        candles: list[dict],
        seq_len: int = 30,
        horizons: list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert raw candle dicts into training arrays (X, y).

        Each candle dict should have: open, high, low, close, volume.
        Targets are binary: 1 if close at horizon > current close, else 0.

        Args:
            candles: List of candle dicts sorted chronologically.
            seq_len: Number of candles per input window.
            horizons: Prediction horizons (default [1, 5, 10]).

        Returns:
            X: (n_samples, seq_len, 5) feature array (OHLCV, normalised).
            y: (n_samples, n_horizons) binary target array.
        """
        horizons = horizons or [1, 5, 10]
        max_horizon = max(horizons)

        if len(candles) < seq_len + max_horizon:
            raise ValueError(
                f"Need at least {seq_len + max_horizon} candles, got {len(candles)}"
            )

        # Extract OHLCV
        raw = np.array([
            [float(c["open"]), float(c["high"]), float(c["low"]),
             float(c["close"]), float(c.get("volume", 0))]
            for c in candles
        ], dtype=np.float32)

        # Normalise each feature column to [0, 1] within the dataset
        mins = raw.min(axis=0, keepdims=True)
        maxs = raw.max(axis=0, keepdims=True)
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0  # avoid div-by-zero
        normalised = (raw - mins) / ranges

        n_total = len(candles)
        samples_X = []
        samples_y = []

        for i in range(seq_len, n_total - max_horizon):
            window = normalised[i - seq_len : i]  # (seq_len, 5)
            current_close = raw[i - 1, 3]  # un-normalised close of last candle in window

            targets = []
            for h in horizons:
                future_close = raw[i - 1 + h, 3]
                targets.append(1.0 if future_close > current_close else 0.0)

            samples_X.append(window)
            samples_y.append(targets)

        X = np.array(samples_X, dtype=np.float32)
        y = np.array(samples_y, dtype=np.float32)
        return X, y

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

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
            "train_losses": self._train_losses,
        }
        # Also save input projections
        input_projs = {}
        for k, v in self._input_projs.items():
            input_projs[str(k)] = v.tolist()
        data["input_projs"] = input_projs

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
            self._train_losses = data.get("train_losses", [])
            # Restore input projections
            for k, v in data.get("input_projs", {}).items():
                self._input_projs[int(k)] = np.array(v, dtype=np.float32)
            logger.info("transformer_loaded", path=str(path), is_trained=self._is_trained)
            return True
        except Exception as e:
            logger.warning("transformer_load_failed", error=str(e))
            return False


# Singleton
transformer_predictor = TemporalFusionPredictor()
