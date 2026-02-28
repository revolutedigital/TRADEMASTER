"""LSTM model for time-series classification: BUY/HOLD/SELL."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.core.logging import get_logger
from app.services.ml.models.base import BaseTradingModel, ModelPrediction, TrainingResult

logger = get_logger(__name__)


class LSTMNetwork(nn.Module):
    """2-layer LSTM with dropout for 3-class classification."""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
        )
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take the last time step
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class LSTMTradingModel(BaseTradingModel):
    """LSTM model wrapper implementing the BaseTradingModel interface."""

    def __init__(self, input_size: int = 0, hidden_size: int = 128, num_layers: int = 2):
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._network: LSTMNetwork | None = None
        self._device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    @property
    def model_type(self) -> str:
        return "lstm"

    @property
    def is_loaded(self) -> bool:
        return self._network is not None

    def predict(self, features: np.ndarray) -> ModelPrediction:
        """Predict on a single sequence.

        Args:
            features: shape (seq_len, n_features)
        """
        if not self._network:
            raise RuntimeError("Model not loaded. Call load() or train() first.")

        self._network.eval()
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0).to(self._device)
            logits = self._network(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

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
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 0.001,
        patience: int = 10,
    ) -> TrainingResult:
        """Train the LSTM model with early stopping."""
        input_size = X_train.shape[2]
        self._input_size = input_size
        self._network = LSTMNetwork(input_size, self._hidden_size, self._num_layers).to(
            self._device
        )

        # Class weights for imbalanced data
        class_counts = np.bincount(y_train, minlength=3).astype(float)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        class_weights /= class_weights.sum()
        weights_tensor = torch.FloatTensor(class_weights).to(self._device)

        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        optimizer = torch.optim.Adam(self._network.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.LongTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), torch.LongTensor(y_val)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        best_val_loss = float("inf")
        best_epoch = 0
        no_improve = 0
        best_state = None

        for epoch in range(epochs):
            # Training
            self._network.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self._device), y_batch.to(self._device)
                optimizer.zero_grad()
                outputs = self._network(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * len(y_batch)
                train_correct += (outputs.argmax(1) == y_batch).sum().item()
                train_total += len(y_batch)

            # Validation
            self._network.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self._device), y_batch.to(self._device)
                    outputs = self._network(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * len(y_batch)
                    val_correct += (outputs.argmax(1) == y_batch).sum().item()
                    val_total += len(y_batch)

            avg_train_loss = train_loss / max(train_total, 1)
            avg_val_loss = val_loss / max(val_total, 1)
            train_acc = train_correct / max(train_total, 1)
            val_acc = val_correct / max(val_total, 1)

            scheduler.step(avg_val_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    "lstm_epoch",
                    epoch=epoch + 1,
                    train_loss=round(avg_train_loss, 4),
                    val_loss=round(avg_val_loss, 4),
                    train_acc=round(train_acc, 4),
                    val_acc=round(val_acc, 4),
                )

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in self._network.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("lstm_early_stop", epoch=epoch + 1, best_epoch=best_epoch)
                    break

        # Restore best weights
        if best_state:
            self._network.load_state_dict(best_state)
            self._network.to(self._device)

        return TrainingResult(
            accuracy=train_acc,
            loss=avg_train_loss,
            val_accuracy=val_acc,
            val_loss=best_val_loss,
            epochs_trained=epoch + 1,
            best_epoch=best_epoch,
        )

    def save(self, path: Path) -> None:
        if not self._network:
            raise RuntimeError("No model to save.")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self._network.state_dict(),
                "input_size": self._input_size,
                "hidden_size": self._hidden_size,
                "num_layers": self._num_layers,
            },
            path,
        )
        logger.info("lstm_model_saved", path=str(path))

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self._device, weights_only=True)
        self._input_size = checkpoint["input_size"]
        self._hidden_size = checkpoint["hidden_size"]
        self._num_layers = checkpoint["num_layers"]
        self._network = LSTMNetwork(
            self._input_size, self._hidden_size, self._num_layers
        ).to(self._device)
        self._network.load_state_dict(checkpoint["model_state"])
        self._network.eval()
        logger.info("lstm_model_loaded", path=str(path))
