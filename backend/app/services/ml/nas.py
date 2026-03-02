"""Neural Architecture Search (NAS) for automated trading model design."""

import math
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


class LayerType(str, Enum):
    DENSE = "dense"
    LSTM = "lstm"
    CONV1D = "conv1d"
    ATTENTION = "attention"
    DROPOUT = "dropout"
    BATCH_NORM = "batch_norm"
    RESIDUAL = "residual"


@dataclass
class LayerConfig:
    """Configuration for a single layer."""
    layer_type: LayerType
    units: int = 64
    activation: str = "relu"
    dropout_rate: float = 0.0
    kernel_size: int = 3
    num_heads: int = 4


@dataclass
class ArchitectureCandidate:
    """A candidate neural network architecture."""
    layers: list[LayerConfig]
    learning_rate: float = 0.001
    batch_size: int = 32
    optimizer: str = "adam"
    score: float = 0.0
    latency_ms: float = 0.0
    params_count: int = 0
    generation: int = 0
    parent_id: str | None = None
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = f"arch_{id(self)}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "layers": [
                {
                    "type": l.layer_type.value,
                    "units": l.units,
                    "activation": l.activation,
                    "dropout_rate": l.dropout_rate,
                }
                for l in self.layers
            ],
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "optimizer": self.optimizer,
            "score": self.score,
            "latency_ms": self.latency_ms,
            "params_count": self.params_count,
            "generation": self.generation,
        }


@dataclass
class NASConfig:
    """Configuration for Neural Architecture Search."""
    input_dim: int = 64
    output_dim: int = 3
    max_layers: int = 8
    min_layers: int = 2
    population_size: int = 20
    n_generations: int = 10
    mutation_rate: float = 0.3
    crossover_rate: float = 0.5
    tournament_size: int = 3
    latency_weight: float = 0.1  # Trade-off between accuracy and speed
    available_units: list[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])
    available_activations: list[str] = field(default_factory=lambda: ["relu", "tanh", "gelu", "swish"])
    available_optimizers: list[str] = field(default_factory=lambda: ["adam", "adamw", "sgd", "rmsprop"])
    available_lr: list[float] = field(default_factory=lambda: [1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    available_batch_sizes: list[int] = field(default_factory=lambda: [16, 32, 64, 128])


class ArchitectureEvaluator:
    """Evaluates candidate architectures using proxy metrics."""

    def __init__(self, config: NASConfig):
        self.config = config

    def estimate_params(self, candidate: ArchitectureCandidate) -> int:
        """Estimate total parameter count."""
        total = 0
        prev_dim = self.config.input_dim

        for layer in candidate.layers:
            if layer.layer_type == LayerType.DENSE:
                total += prev_dim * layer.units + layer.units
                prev_dim = layer.units
            elif layer.layer_type == LayerType.LSTM:
                total += 4 * (prev_dim * layer.units + layer.units * layer.units + layer.units)
                prev_dim = layer.units
            elif layer.layer_type == LayerType.CONV1D:
                total += layer.kernel_size * prev_dim * layer.units + layer.units
                prev_dim = layer.units
            elif layer.layer_type == LayerType.ATTENTION:
                total += 3 * prev_dim * prev_dim + prev_dim * prev_dim
            elif layer.layer_type in (LayerType.DROPOUT, LayerType.BATCH_NORM):
                if layer.layer_type == LayerType.BATCH_NORM:
                    total += 2 * prev_dim
            elif layer.layer_type == LayerType.RESIDUAL:
                total += prev_dim * layer.units * 2 + layer.units * 2
                prev_dim = layer.units

        # Output layer
        total += prev_dim * self.config.output_dim + self.config.output_dim
        return total

    def estimate_latency(self, candidate: ArchitectureCandidate) -> float:
        """Estimate inference latency in milliseconds."""
        params = self.estimate_params(candidate)
        base_latency = 0.5  # Base overhead ms

        for layer in candidate.layers:
            if layer.layer_type == LayerType.DENSE:
                base_latency += 0.01 * layer.units / 64
            elif layer.layer_type == LayerType.LSTM:
                base_latency += 0.1 * layer.units / 64
            elif layer.layer_type == LayerType.CONV1D:
                base_latency += 0.05 * layer.units / 64
            elif layer.layer_type == LayerType.ATTENTION:
                base_latency += 0.15 * (layer.num_heads / 4)
            elif layer.layer_type == LayerType.RESIDUAL:
                base_latency += 0.02 * layer.units / 64

        return base_latency

    def evaluate(self, candidate: ArchitectureCandidate,
                 X_val: np.ndarray | None = None,
                 y_val: np.ndarray | None = None) -> float:
        """
        Evaluate architecture using proxy metrics + optional validation data.

        Score combines: accuracy proxy, latency efficiency, architecture quality.
        """
        params = self.estimate_params(candidate)
        latency = self.estimate_latency(candidate)
        candidate.params_count = params
        candidate.latency_ms = latency

        # Proxy score based on architecture heuristics
        score = 0.0

        # Depth bonus (deeper is generally better, up to a point)
        n_layers = len([l for l in candidate.layers if l.layer_type not in (LayerType.DROPOUT, LayerType.BATCH_NORM)])
        depth_score = min(n_layers / 4.0, 1.0)  # Optimal around 4 layers
        if n_layers > 6:
            depth_score -= (n_layers - 6) * 0.1  # Penalize very deep
        score += depth_score * 0.2

        # Width bonus (wider layers capture more patterns)
        max_units = max((l.units for l in candidate.layers if l.layer_type != LayerType.DROPOUT), default=64)
        width_score = min(max_units / 256.0, 1.0)
        score += width_score * 0.15

        # Architecture diversity bonus
        layer_types = set(l.layer_type for l in candidate.layers)
        diversity_score = len(layer_types) / len(LayerType)
        score += diversity_score * 0.15

        # Has regularization (dropout, batch norm)
        has_dropout = any(l.layer_type == LayerType.DROPOUT for l in candidate.layers)
        has_bn = any(l.layer_type == LayerType.BATCH_NORM for l in candidate.layers)
        score += 0.1 if has_dropout else 0
        score += 0.05 if has_bn else 0

        # Has attention (good for time series)
        has_attention = any(l.layer_type == LayerType.ATTENTION for l in candidate.layers)
        score += 0.1 if has_attention else 0

        # Has LSTM (good for sequential data)
        has_lstm = any(l.layer_type == LayerType.LSTM for l in candidate.layers)
        score += 0.1 if has_lstm else 0

        # Residual connections
        has_residual = any(l.layer_type == LayerType.RESIDUAL for l in candidate.layers)
        score += 0.05 if has_residual else 0

        # Learning rate (moderate is best)
        if 1e-4 <= candidate.learning_rate <= 3e-3:
            score += 0.05

        # Latency penalty
        score -= latency * self.config.latency_weight * 0.01

        # Parameter efficiency
        if params < 100_000:
            score += 0.05  # Efficient model bonus
        elif params > 10_000_000:
            score -= 0.1  # Too large penalty

        # If validation data provided, do a quick proxy evaluation
        if X_val is not None and y_val is not None:
            proxy_score = self._proxy_evaluate(candidate, X_val, y_val)
            score = score * 0.4 + proxy_score * 0.6

        candidate.score = max(0.0, min(1.0, score))
        return candidate.score

    def _proxy_evaluate(self, candidate: ArchitectureCandidate,
                       X: np.ndarray, y: np.ndarray) -> float:
        """Quick proxy evaluation using simplified forward pass."""
        rng = np.random.RandomState(42)
        prev_dim = X.shape[-1] if X.ndim > 1 else 1
        h = X.copy()

        if h.ndim == 1:
            h = h.reshape(-1, 1)

        for layer in candidate.layers:
            if layer.layer_type in (LayerType.DENSE, LayerType.RESIDUAL):
                w = rng.randn(prev_dim, layer.units) * math.sqrt(2.0 / prev_dim)
                h = h @ w[:h.shape[-1], :] if h.shape[-1] <= prev_dim else h[:, :prev_dim] @ w
                h = np.tanh(h)  # Use tanh for stability
                prev_dim = layer.units
            elif layer.layer_type == LayerType.DROPOUT:
                mask = rng.binomial(1, 1 - layer.dropout_rate, h.shape)
                h = h * mask
            elif layer.layer_type == LayerType.BATCH_NORM:
                h = (h - h.mean(axis=0)) / (h.std(axis=0) + 1e-8)
            # Skip LSTM/Conv1D/Attention for proxy (use dense approximation)
            elif layer.layer_type in (LayerType.LSTM, LayerType.CONV1D, LayerType.ATTENTION):
                w = rng.randn(prev_dim, layer.units) * math.sqrt(2.0 / prev_dim)
                h = h @ w[:h.shape[-1], :] if h.shape[-1] <= prev_dim else h[:, :prev_dim] @ w
                h = np.tanh(h)
                prev_dim = layer.units

        # Output layer
        w_out = rng.randn(prev_dim, self.config.output_dim) * math.sqrt(2.0 / prev_dim)
        output = h @ w_out[:h.shape[-1], :] if h.shape[-1] <= prev_dim else h[:, :prev_dim] @ w_out

        # Score based on output distribution quality
        if output.ndim > 1:
            output_std = np.std(output, axis=0).mean()
            output_range = (np.max(output) - np.min(output))
        else:
            output_std = float(np.std(output))
            output_range = float(np.max(output) - np.min(output))

        # Good models produce diverse, well-separated outputs
        distribution_quality = min(output_std * 2, 1.0) if output_std > 0 else 0.0
        separation_quality = min(output_range / 5.0, 1.0)

        return (distribution_quality + separation_quality) / 2


class NeuralArchitectureSearch:
    """
    Evolutionary Neural Architecture Search for trading models.

    Uses genetic algorithm to evolve network architectures:
    - Tournament selection
    - Crossover (layer swapping)
    - Mutation (add/remove/modify layers)
    """

    def __init__(self, config: NASConfig | None = None):
        self.config = config or NASConfig()
        self.evaluator = ArchitectureEvaluator(self.config)
        self.population: list[ArchitectureCandidate] = []
        self.hall_of_fame: list[ArchitectureCandidate] = []
        self.search_history: list[dict] = []
        self._rng = np.random.RandomState(42)

        logger.info("nas_initialized", population_size=self.config.population_size,
                    generations=self.config.n_generations)

    def _random_layer(self) -> LayerConfig:
        """Generate a random layer configuration."""
        layer_type = self._rng.choice(list(LayerType))
        return LayerConfig(
            layer_type=layer_type,
            units=int(self._rng.choice(self.config.available_units)),
            activation=self._rng.choice(self.config.available_activations),
            dropout_rate=float(self._rng.uniform(0.0, 0.5)) if layer_type == LayerType.DROPOUT else 0.0,
            kernel_size=int(self._rng.choice([3, 5, 7])) if layer_type == LayerType.CONV1D else 3,
            num_heads=int(self._rng.choice([2, 4, 8])) if layer_type == LayerType.ATTENTION else 4,
        )

    def _random_architecture(self, generation: int = 0) -> ArchitectureCandidate:
        """Generate a random architecture candidate."""
        n_layers = self._rng.randint(self.config.min_layers, self.config.max_layers + 1)
        layers = [self._random_layer() for _ in range(n_layers)]

        # Ensure at least one computational layer
        if not any(l.layer_type in (LayerType.DENSE, LayerType.LSTM, LayerType.CONV1D) for l in layers):
            layers[0] = LayerConfig(layer_type=LayerType.DENSE,
                                    units=int(self._rng.choice(self.config.available_units)))

        return ArchitectureCandidate(
            layers=layers,
            learning_rate=float(self._rng.choice(self.config.available_lr)),
            batch_size=int(self._rng.choice(self.config.available_batch_sizes)),
            optimizer=self._rng.choice(self.config.available_optimizers),
            generation=generation,
            id=f"arch_g{generation}_{self._rng.randint(100000)}",
        )

    def _tournament_select(self) -> ArchitectureCandidate:
        """Select a candidate using tournament selection."""
        tournament = self._rng.choice(self.population, size=min(self.config.tournament_size, len(self.population)),
                                       replace=False)
        return max(tournament, key=lambda c: c.score)

    def _crossover(self, parent1: ArchitectureCandidate,
                   parent2: ArchitectureCandidate,
                   generation: int) -> ArchitectureCandidate:
        """Create child by crossing over two parents."""
        # Single-point crossover on layers
        cut1 = self._rng.randint(1, max(2, len(parent1.layers)))
        cut2 = self._rng.randint(1, max(2, len(parent2.layers)))

        child_layers = parent1.layers[:cut1] + parent2.layers[cut2:]

        # Trim if too long
        if len(child_layers) > self.config.max_layers:
            child_layers = child_layers[:self.config.max_layers]
        # Pad if too short
        while len(child_layers) < self.config.min_layers:
            child_layers.append(self._random_layer())

        return ArchitectureCandidate(
            layers=child_layers,
            learning_rate=self._rng.choice([parent1.learning_rate, parent2.learning_rate]),
            batch_size=self._rng.choice([parent1.batch_size, parent2.batch_size]),
            optimizer=self._rng.choice([parent1.optimizer, parent2.optimizer]),
            generation=generation,
            parent_id=f"{parent1.id}+{parent2.id}",
            id=f"arch_g{generation}_{self._rng.randint(100000)}",
        )

    def _mutate(self, candidate: ArchitectureCandidate) -> ArchitectureCandidate:
        """Mutate a candidate architecture."""
        layers = [LayerConfig(**l.__dict__) for l in candidate.layers]

        mutation_type = self._rng.choice(["add", "remove", "modify", "swap", "hyperparams"])

        if mutation_type == "add" and len(layers) < self.config.max_layers:
            pos = self._rng.randint(0, len(layers) + 1)
            layers.insert(pos, self._random_layer())

        elif mutation_type == "remove" and len(layers) > self.config.min_layers:
            pos = self._rng.randint(0, len(layers))
            layers.pop(pos)

        elif mutation_type == "modify" and layers:
            pos = self._rng.randint(0, len(layers))
            what = self._rng.choice(["type", "units", "activation"])
            if what == "type":
                layers[pos].layer_type = self._rng.choice(list(LayerType))
            elif what == "units":
                layers[pos].units = int(self._rng.choice(self.config.available_units))
            elif what == "activation":
                layers[pos].activation = self._rng.choice(self.config.available_activations)

        elif mutation_type == "swap" and len(layers) > 1:
            i, j = self._rng.choice(len(layers), size=2, replace=False)
            layers[i], layers[j] = layers[j], layers[i]

        elif mutation_type == "hyperparams":
            candidate.learning_rate = float(self._rng.choice(self.config.available_lr))
            candidate.batch_size = int(self._rng.choice(self.config.available_batch_sizes))
            candidate.optimizer = self._rng.choice(self.config.available_optimizers)

        candidate.layers = layers
        return candidate

    def search(self, X_val: np.ndarray | None = None,
               y_val: np.ndarray | None = None) -> dict:
        """
        Run the full NAS evolutionary search.

        Returns the best architecture found and search statistics.
        """
        start_time = time.time()

        # Initialize population
        self.population = [self._random_architecture(0) for _ in range(self.config.population_size)]

        # Evaluate initial population
        for candidate in self.population:
            self.evaluator.evaluate(candidate, X_val, y_val)

        self.population.sort(key=lambda c: c.score, reverse=True)
        self.hall_of_fame = [self.population[0]]

        logger.info("nas_initial_population", best_score=self.population[0].score,
                    avg_score=np.mean([c.score for c in self.population]))

        for gen in range(1, self.config.n_generations + 1):
            new_population = []

            # Elitism: keep top 2
            new_population.extend(self.population[:2])

            while len(new_population) < self.config.population_size:
                if self._rng.random() < self.config.crossover_rate:
                    # Crossover
                    parent1 = self._tournament_select()
                    parent2 = self._tournament_select()
                    child = self._crossover(parent1, parent2, gen)
                else:
                    # Clone + mutate
                    parent = self._tournament_select()
                    child = ArchitectureCandidate(
                        layers=[LayerConfig(**l.__dict__) for l in parent.layers],
                        learning_rate=parent.learning_rate,
                        batch_size=parent.batch_size,
                        optimizer=parent.optimizer,
                        generation=gen,
                        parent_id=parent.id,
                        id=f"arch_g{gen}_{self._rng.randint(100000)}",
                    )

                # Mutation
                if self._rng.random() < self.config.mutation_rate:
                    child = self._mutate(child)

                # Evaluate
                self.evaluator.evaluate(child, X_val, y_val)
                new_population.append(child)

            self.population = sorted(new_population, key=lambda c: c.score, reverse=True)

            # Update hall of fame
            if self.population[0].score > self.hall_of_fame[0].score:
                self.hall_of_fame.insert(0, self.population[0])
                self.hall_of_fame = self.hall_of_fame[:10]

            gen_stats = {
                "generation": gen,
                "best_score": float(self.population[0].score),
                "avg_score": float(np.mean([c.score for c in self.population])),
                "best_params": self.population[0].params_count,
                "best_latency_ms": self.population[0].latency_ms,
                "best_n_layers": len(self.population[0].layers),
            }
            self.search_history.append(gen_stats)

            if gen % 2 == 0:
                logger.info("nas_generation", **gen_stats)

        elapsed = time.time() - start_time
        best = self.hall_of_fame[0]

        result = {
            "best_architecture": best.to_dict(),
            "search_stats": {
                "total_generations": self.config.n_generations,
                "total_evaluations": self.config.n_generations * self.config.population_size,
                "elapsed_seconds": round(elapsed, 2),
                "best_score": float(best.score),
                "best_params": best.params_count,
                "best_latency_ms": round(best.latency_ms, 3),
            },
            "top_5_architectures": [c.to_dict() for c in self.hall_of_fame[:5]],
            "convergence_history": self.search_history,
            "pareto_front": self._compute_pareto_front(),
        }

        logger.info("nas_search_complete", best_score=best.score,
                    best_params=best.params_count, elapsed_s=round(elapsed, 2))
        return result

    def _compute_pareto_front(self) -> list[dict]:
        """Compute Pareto front: best accuracy-latency trade-offs."""
        candidates = sorted(self.population, key=lambda c: c.score, reverse=True)
        pareto = []
        min_latency = float("inf")

        for c in candidates:
            if c.latency_ms < min_latency:
                pareto.append({
                    "id": c.id,
                    "score": c.score,
                    "latency_ms": c.latency_ms,
                    "params": c.params_count,
                    "n_layers": len(c.layers),
                })
                min_latency = c.latency_ms

        return pareto

    def suggest_architecture(self, priority: str = "balanced") -> dict:
        """Suggest the best architecture based on priority."""
        if not self.hall_of_fame:
            return {"error": "Run search() first"}

        if priority == "accuracy":
            best = max(self.hall_of_fame, key=lambda c: c.score)
        elif priority == "speed":
            best = min(self.hall_of_fame, key=lambda c: c.latency_ms)
        elif priority == "compact":
            best = min(self.hall_of_fame, key=lambda c: c.params_count)
        else:  # balanced
            best = max(self.hall_of_fame,
                      key=lambda c: c.score - c.latency_ms * self.config.latency_weight * 0.01)

        return {
            "priority": priority,
            "architecture": best.to_dict(),
            "recommendation": f"Architecture {best.id} with {len(best.layers)} layers, "
                            f"{best.params_count} params, {best.latency_ms:.1f}ms latency",
        }


# Module-level instance
nas_searcher = NeuralArchitectureSearch()
