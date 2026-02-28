"""Temporal Fusion Transformer (simplified) for multi-horizon trading prediction."""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from app.core.logging import get_logger
from app.services.ml.models.base import BaseTradingModel, ModelPrediction, TrainingResult

logger = get_logger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position awareness."""

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerNetwork(nn.Module):
    """Transformer encoder for time series classification.

    Architecture:
        Input -> Linear projection -> Positional Encoding ->
        N x TransformerEncoderLayer -> Global Average Pool -> FC -> 3-class output
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        num_classes: int = 3,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        # Global average pooling over time
        x = x.mean(dim=1)
        return self.classifier(x)


class TransformerTradingModel(BaseTradingModel):
    """Transformer-based trading model.

    Uses self-attention to capture long-range dependencies in
    price sequences, outperforming LSTM on irregular patterns.
    """

    model_type = "transformer"

    def __init__(self):
        self._model: TransformerNetwork | None = None
        self._is_loaded = False
        self._device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 80,
        batch_size: int = 64,
        patience: int = 10,
        lr: float = 1e-4,
    ) -> TrainingResult:
        input_dim = X_train.shape[2]
        self._model = TransformerNetwork(input_dim=input_dim).to(self._device)

        # Class weights for imbalance
        classes, counts = np.unique(y_train, return_counts=True)
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(classes)
        class_weights = torch.FloatTensor(weights).to(self._device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

        X_t = torch.FloatTensor(X_train).to(self._device)
        y_t = torch.LongTensor(y_train).to(self._device)
        X_v = torch.FloatTensor(X_val).to(self._device)
        y_v = torch.LongTensor(y_val).to(self._device)

        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            self._model.train()
            indices = torch.randperm(len(X_t))

            epoch_loss = 0
            n_batches = 0

            for i in range(0, len(X_t), batch_size):
                batch_idx = indices[i : i + batch_size]
                xb = X_t[batch_idx]
                yb = y_t[batch_idx]

                optimizer.zero_grad()
                output = self._model(xb)
                loss = criterion(output, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            # Validation
            self._model.eval()
            with torch.no_grad():
                val_output = self._model(X_v)
                val_loss = criterion(val_output, y_v).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Restore best
        if best_state:
            self._model.load_state_dict(best_state)
            self._model.to(self._device)

        self._is_loaded = True

        # Calculate accuracies
        self._model.eval()
        with torch.no_grad():
            train_pred = self._model(X_t).argmax(dim=1).cpu().numpy()
            val_pred = self._model(X_v).argmax(dim=1).cpu().numpy()

        train_acc = float(np.mean(train_pred == y_train))
        val_acc = float(np.mean(val_pred == y_val))

        logger.info(
            "transformer_trained",
            train_acc=round(train_acc, 4),
            val_acc=round(val_acc, 4),
            best_epoch=best_epoch,
        )

        return TrainingResult(
            accuracy=train_acc,
            val_accuracy=val_acc,
            loss=epoch_loss / max(n_batches, 1),
            val_loss=best_val_loss,
            epochs_trained=epoch + 1,
            best_epoch=best_epoch,
        )

    def predict(self, features: np.ndarray) -> ModelPrediction:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        self._model.eval()

        if features.ndim == 2:
            features = features[np.newaxis, :]

        with torch.no_grad():
            x = torch.FloatTensor(features).to(self._device)
            logits = self._model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        action = int(np.argmax(probs))
        signal = float(probs[2] - probs[0])

        return ModelPrediction(
            action=action,
            probabilities=probs,
            confidence=float(probs[action]),
            signal_strength=signal,
        )

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("No model to save")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self._model.state_dict(),
                "input_dim": self._model.input_projection.in_features,
            },
            path,
        )
        logger.info("transformer_saved", path=str(path))

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self._device, weights_only=True)
        self._model = TransformerNetwork(
            input_dim=checkpoint["input_dim"]
        ).to(self._device)
        self._model.load_state_dict(checkpoint["model_state"])
        self._model.eval()
        self._is_loaded = True
        logger.info("transformer_loaded", path=str(path))
