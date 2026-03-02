"""Federated learning framework for privacy-preserving model training."""

import copy
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FederatedConfig:
    """Configuration for federated learning."""
    n_rounds: int = 10
    min_clients: int = 2
    fraction_fit: float = 0.8
    fraction_evaluate: float = 0.5
    learning_rate: float = 0.01
    aggregation_strategy: str = "fedavg"  # fedavg, fedprox, fedyogi


@dataclass
class ClientUpdate:
    """Update from a single federated client."""
    client_id: str
    weights: list[np.ndarray]
    n_samples: int
    metrics: dict = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


class FederatedAggregator:
    """
    Federated learning aggregator for privacy-preserving model training.

    Supports:
    - FedAvg: Federated Averaging (weighted by sample count)
    - FedProx: Proximal term for heterogeneous clients
    - FedYogi: Adaptive federated optimization

    Use case: Train trading models across multiple users without
    sharing raw portfolio data or trading history.
    """

    def __init__(self, config: FederatedConfig | None = None):
        self.config = config or FederatedConfig()
        self._global_weights: list[np.ndarray] | None = None
        self._round: int = 0
        self._history: list[dict] = []

        logger.info("federated_aggregator_initialized",
                    strategy=self.config.aggregation_strategy)

    def initialize_global_model(self, weight_shapes: list[tuple[int, ...]]) -> None:
        """Initialize global model weights."""
        self._global_weights = [
            np.random.randn(*shape) * 0.01 for shape in weight_shapes
        ]
        logger.info("global_model_initialized", n_layers=len(weight_shapes))

    def get_global_weights(self) -> list[np.ndarray] | None:
        """Get current global model weights for distribution to clients."""
        return self._global_weights

    def aggregate(self, client_updates: list[ClientUpdate]) -> dict:
        """
        Aggregate client updates into new global model.

        Args:
            client_updates: List of updates from participating clients

        Returns:
            Aggregation results including new global metrics
        """
        if len(client_updates) < self.config.min_clients:
            return {
                "status": "insufficient_clients",
                "received": len(client_updates),
                "required": self.config.min_clients,
            }

        self._round += 1

        if self.config.aggregation_strategy == "fedavg":
            new_weights = self._fedavg(client_updates)
        elif self.config.aggregation_strategy == "fedprox":
            new_weights = self._fedprox(client_updates)
        elif self.config.aggregation_strategy == "fedyogi":
            new_weights = self._fedyogi(client_updates)
        else:
            new_weights = self._fedavg(client_updates)

        self._global_weights = new_weights

        # Aggregate metrics
        total_samples = sum(u.n_samples for u in client_updates)
        avg_metrics = {}
        metric_keys = set()
        for u in client_updates:
            metric_keys.update(u.metrics.keys())

        for key in metric_keys:
            values = [u.metrics.get(key, 0) * u.n_samples for u in client_updates if key in u.metrics]
            avg_metrics[key] = sum(values) / total_samples if total_samples > 0 else 0

        round_result = {
            "round": self._round,
            "n_clients": len(client_updates),
            "total_samples": total_samples,
            "strategy": self.config.aggregation_strategy,
            "metrics": avg_metrics,
            "timestamp": datetime.utcnow().isoformat(),
        }

        self._history.append(round_result)
        logger.info("federated_round_complete", **round_result)

        return round_result

    def _fedavg(self, updates: list[ClientUpdate]) -> list[np.ndarray]:
        """Federated Averaging: weighted average by number of samples."""
        total_samples = sum(u.n_samples for u in updates)
        if total_samples == 0:
            return self._global_weights or []

        n_layers = len(updates[0].weights)
        new_weights = []

        for layer_idx in range(n_layers):
            weighted_sum = np.zeros_like(updates[0].weights[layer_idx])
            for update in updates:
                weight_factor = update.n_samples / total_samples
                weighted_sum += update.weights[layer_idx] * weight_factor
            new_weights.append(weighted_sum)

        return new_weights

    def _fedprox(self, updates: list[ClientUpdate], mu: float = 0.01) -> list[np.ndarray]:
        """FedProx: FedAvg with proximal term for heterogeneous clients."""
        avg_weights = self._fedavg(updates)

        if self._global_weights is None:
            return avg_weights

        # Apply proximal regularization toward global model
        new_weights = []
        for avg_w, global_w in zip(avg_weights, self._global_weights):
            proximal = avg_w - mu * (avg_w - global_w)
            new_weights.append(proximal)

        return new_weights

    def _fedyogi(self, updates: list[ClientUpdate],
                 beta1: float = 0.9, beta2: float = 0.99,
                 tau: float = 1e-3) -> list[np.ndarray]:
        """FedYogi: Adaptive federated optimization."""
        avg_weights = self._fedavg(updates)

        if self._global_weights is None:
            return avg_weights

        new_weights = []
        for avg_w, global_w in zip(avg_weights, self._global_weights):
            delta = avg_w - global_w
            # Adaptive learning rate (simplified Yogi update)
            v = delta ** 2
            adaptive_lr = self.config.learning_rate / (np.sqrt(v) + tau)
            new_w = global_w + adaptive_lr * delta
            new_weights.append(new_w)

        return new_weights

    def get_training_status(self) -> dict:
        """Get current federated training status."""
        return {
            "current_round": self._round,
            "total_rounds": self.config.n_rounds,
            "strategy": self.config.aggregation_strategy,
            "is_complete": self._round >= self.config.n_rounds,
            "model_initialized": self._global_weights is not None,
            "history": self._history[-10:],  # Last 10 rounds
        }


class FederatedClient:
    """
    Simulated federated learning client.

    Each client represents a user with private trading data.
    Trains local model and sends only weight updates (not raw data).
    """

    def __init__(self, client_id: str, local_data: np.ndarray | None = None,
                 local_labels: np.ndarray | None = None):
        self.client_id = client_id
        self.local_data = local_data
        self.local_labels = local_labels
        self._local_weights: list[np.ndarray] | None = None

    def receive_global_weights(self, weights: list[np.ndarray]) -> None:
        """Receive and store global model weights."""
        self._local_weights = [w.copy() for w in weights]

    def train_local(self, epochs: int = 5, lr: float = 0.01) -> ClientUpdate:
        """Train local model on private data."""
        if self._local_weights is None or self.local_data is None:
            return ClientUpdate(
                client_id=self.client_id,
                weights=[],
                n_samples=0,
                metrics={"error": "no_data"},
            )

        weights = [w.copy() for w in self._local_weights]
        n_samples = len(self.local_data)

        # Simplified local SGD training
        for epoch in range(epochs):
            for i in range(0, n_samples, 32):
                batch_x = self.local_data[i:i+32]
                batch_y = self.local_labels[i:i+32] if self.local_labels is not None else None

                # Forward pass through layers
                h = batch_x
                for j, w in enumerate(weights):
                    if h.shape[-1] != w.shape[0]:
                        continue
                    h = h @ w
                    if j < len(weights) - 1:
                        h = np.tanh(h)

                # Simplified gradient update
                if batch_y is not None and h.shape == batch_y.shape:
                    error = h - batch_y
                    for j in range(len(weights)):
                        grad = np.random.randn(*weights[j].shape) * np.mean(error ** 2) * 0.01
                        weights[j] -= lr * grad

        # Compute local metrics
        metrics = {"loss": 0.0, "accuracy": 0.0}
        if self.local_labels is not None:
            h = self.local_data
            for j, w in enumerate(weights):
                if h.shape[-1] != w.shape[0]:
                    break
                h = h @ w
                if j < len(weights) - 1:
                    h = np.tanh(h)
            if h.shape == self.local_labels.shape:
                metrics["loss"] = float(np.mean((h - self.local_labels) ** 2))
                predictions = (h > 0.5).astype(float)
                metrics["accuracy"] = float(np.mean(predictions == self.local_labels))

        return ClientUpdate(
            client_id=self.client_id,
            weights=weights,
            n_samples=n_samples,
            metrics=metrics,
        )


# Module-level instances
federated_aggregator = FederatedAggregator()
