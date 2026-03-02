"""Synthetic market data generation for training augmentation and stress testing."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


class MarketRegime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    CRASH = "crash"
    RECOVERY = "recovery"
    BUBBLE = "bubble"


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    base_price: float = 50000.0
    n_steps: int = 1000
    dt: float = 1 / 365  # Daily frequency
    seed: int | None = None


class GeometricBrownianMotion:
    """Generate price paths using GBM."""

    def generate(self, config: SyntheticConfig, mu: float = 0.05,
                 sigma: float = 0.3) -> np.ndarray:
        rng = np.random.RandomState(config.seed)
        prices = np.zeros(config.n_steps)
        prices[0] = config.base_price

        for t in range(1, config.n_steps):
            z = rng.standard_normal()
            prices[t] = prices[t - 1] * np.exp(
                (mu - 0.5 * sigma ** 2) * config.dt + sigma * np.sqrt(config.dt) * z
            )

        return prices


class JumpDiffusion:
    """Merton's jump-diffusion model for fat tails and sudden moves."""

    def generate(self, config: SyntheticConfig, mu: float = 0.05, sigma: float = 0.3,
                 jump_intensity: float = 5.0, jump_mean: float = -0.02,
                 jump_std: float = 0.05) -> np.ndarray:
        rng = np.random.RandomState(config.seed)
        prices = np.zeros(config.n_steps)
        prices[0] = config.base_price

        for t in range(1, config.n_steps):
            z = rng.standard_normal()
            # Number of jumps (Poisson)
            n_jumps = rng.poisson(jump_intensity * config.dt)
            jump_component = sum(rng.normal(jump_mean, jump_std) for _ in range(n_jumps))

            prices[t] = prices[t - 1] * np.exp(
                (mu - 0.5 * sigma ** 2) * config.dt +
                sigma * np.sqrt(config.dt) * z +
                jump_component
            )

        return prices


class RegimeSwitch:
    """Regime-switching model with Markov transitions."""

    REGIME_PARAMS = {
        MarketRegime.BULL: {"mu": 0.15, "sigma": 0.20},
        MarketRegime.BEAR: {"mu": -0.20, "sigma": 0.35},
        MarketRegime.SIDEWAYS: {"mu": 0.0, "sigma": 0.15},
        MarketRegime.CRASH: {"mu": -0.60, "sigma": 0.80},
        MarketRegime.RECOVERY: {"mu": 0.30, "sigma": 0.40},
        MarketRegime.BUBBLE: {"mu": 0.50, "sigma": 0.50},
    }

    TRANSITION_MATRIX = {
        MarketRegime.BULL: {MarketRegime.BULL: 0.85, MarketRegime.SIDEWAYS: 0.08,
                            MarketRegime.BEAR: 0.05, MarketRegime.CRASH: 0.02},
        MarketRegime.BEAR: {MarketRegime.BEAR: 0.80, MarketRegime.SIDEWAYS: 0.10,
                            MarketRegime.RECOVERY: 0.08, MarketRegime.CRASH: 0.02},
        MarketRegime.SIDEWAYS: {MarketRegime.SIDEWAYS: 0.70, MarketRegime.BULL: 0.15,
                                 MarketRegime.BEAR: 0.12, MarketRegime.CRASH: 0.03},
        MarketRegime.CRASH: {MarketRegime.CRASH: 0.30, MarketRegime.BEAR: 0.40,
                              MarketRegime.RECOVERY: 0.30},
        MarketRegime.RECOVERY: {MarketRegime.RECOVERY: 0.60, MarketRegime.BULL: 0.25,
                                 MarketRegime.SIDEWAYS: 0.15},
        MarketRegime.BUBBLE: {MarketRegime.BUBBLE: 0.75, MarketRegime.CRASH: 0.15,
                               MarketRegime.BULL: 0.10},
    }

    def generate(self, config: SyntheticConfig,
                 initial_regime: MarketRegime = MarketRegime.SIDEWAYS) -> tuple[np.ndarray, list[str]]:
        rng = np.random.RandomState(config.seed)
        prices = np.zeros(config.n_steps)
        regimes = []
        prices[0] = config.base_price
        current_regime = initial_regime

        for t in range(1, config.n_steps):
            # Transition
            transitions = self.TRANSITION_MATRIX.get(current_regime, {current_regime: 1.0})
            states = list(transitions.keys())
            probs = list(transitions.values())
            total = sum(probs)
            probs = [p / total for p in probs]
            current_regime = rng.choice(states, p=probs)
            regimes.append(current_regime.value)

            # Generate price using regime parameters
            params = self.REGIME_PARAMS[current_regime]
            z = rng.standard_normal()
            prices[t] = prices[t - 1] * np.exp(
                (params["mu"] - 0.5 * params["sigma"] ** 2) * config.dt +
                params["sigma"] * np.sqrt(config.dt) * z
            )

        return prices, regimes


class HistoricalScenarioReplay:
    """Replay historical crash scenarios with synthetic modifications."""

    SCENARIOS = {
        "covid_crash_2020": {
            "description": "COVID-19 market crash (March 2020)",
            "phases": [
                {"days": 10, "return_pct": -0.10, "volatility": 0.4},   # Initial fear
                {"days": 5, "return_pct": -0.35, "volatility": 0.9},    # Panic selling
                {"days": 3, "return_pct": -0.15, "volatility": 1.0},    # Capitulation
                {"days": 15, "return_pct": 0.25, "volatility": 0.6},    # Dead cat bounce
                {"days": 60, "return_pct": 0.80, "volatility": 0.4},    # Recovery
            ],
        },
        "luna_crash_2022": {
            "description": "Terra/Luna ecosystem collapse (May 2022)",
            "phases": [
                {"days": 3, "return_pct": -0.20, "volatility": 0.5},    # UST depeg starts
                {"days": 2, "return_pct": -0.50, "volatility": 1.2},    # Bank run
                {"days": 1, "return_pct": -0.90, "volatility": 2.0},    # Hyperinflation
                {"days": 30, "return_pct": -0.30, "volatility": 0.6},   # Contagion
                {"days": 90, "return_pct": 0.10, "volatility": 0.3},    # Slow recovery
            ],
        },
        "ftx_collapse_2022": {
            "description": "FTX exchange collapse (November 2022)",
            "phases": [
                {"days": 5, "return_pct": -0.15, "volatility": 0.5},    # Alameda revelations
                {"days": 3, "return_pct": -0.25, "volatility": 0.8},    # Bank run on FTX
                {"days": 2, "return_pct": -0.10, "volatility": 0.6},    # Bankruptcy filing
                {"days": 20, "return_pct": -0.15, "volatility": 0.4},   # Contagion fears
                {"days": 60, "return_pct": 0.20, "volatility": 0.3},    # Gradual recovery
            ],
        },
        "flash_crash": {
            "description": "Generic flash crash (minutes-scale)",
            "phases": [
                {"days": 1, "return_pct": -0.30, "volatility": 1.5},    # Flash crash
                {"days": 1, "return_pct": 0.20, "volatility": 1.0},     # Immediate recovery
                {"days": 5, "return_pct": 0.05, "volatility": 0.5},     # Stabilization
            ],
        },
        "black_swan": {
            "description": "Extreme black swan event",
            "phases": [
                {"days": 1, "return_pct": -0.50, "volatility": 2.0},    # Shock
                {"days": 7, "return_pct": -0.30, "volatility": 1.5},    # Cascading failures
                {"days": 14, "return_pct": -0.10, "volatility": 0.8},   # Extended decline
                {"days": 30, "return_pct": 0.15, "volatility": 0.5},    # Tentative recovery
                {"days": 90, "return_pct": 0.40, "volatility": 0.4},    # Full recovery
            ],
        },
    }

    def replay(self, scenario_name: str, config: SyntheticConfig) -> tuple[np.ndarray, dict]:
        """Replay a historical scenario with randomized noise."""
        if scenario_name not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(self.SCENARIOS.keys())}")

        scenario = self.SCENARIOS[scenario_name]
        rng = np.random.RandomState(config.seed)

        prices = [config.base_price]
        phase_markers = []

        for phase in scenario["phases"]:
            n_steps = phase["days"]
            target_return = phase["return_pct"]
            vol = phase["volatility"]

            daily_drift = target_return / n_steps
            for _ in range(n_steps):
                z = rng.standard_normal()
                new_price = prices[-1] * np.exp(daily_drift + vol * np.sqrt(config.dt) * z)
                prices.append(max(new_price, 0.01))

            phase_markers.append({
                "start_idx": len(prices) - n_steps,
                "end_idx": len(prices) - 1,
                "days": n_steps,
                "expected_return": target_return,
                "actual_return": float((prices[-1] / prices[-n_steps - 1]) - 1),
            })

        return np.array(prices), {
            "scenario": scenario_name,
            "description": scenario["description"],
            "phases": phase_markers,
            "total_return": float((prices[-1] / prices[0]) - 1),
        }


class SyntheticDataGenerator:
    """
    Unified synthetic data generator for market data augmentation.

    Supports:
    - Geometric Brownian Motion (standard)
    - Jump Diffusion (fat tails)
    - Regime Switching (market cycles)
    - Historical Scenario Replay (stress testing)
    - OHLCV generation from price paths
    """

    def __init__(self):
        self.gbm = GeometricBrownianMotion()
        self.jump_diffusion = JumpDiffusion()
        self.regime_switch = RegimeSwitch()
        self.scenario_replay = HistoricalScenarioReplay()

    def generate_price_path(self, method: str = "gbm",
                            config: SyntheticConfig | None = None,
                            **kwargs) -> np.ndarray:
        """Generate a synthetic price path."""
        config = config or SyntheticConfig()

        generators = {
            "gbm": lambda: self.gbm.generate(config, **kwargs),
            "jump_diffusion": lambda: self.jump_diffusion.generate(config, **kwargs),
        }

        if method in generators:
            return generators[method]()
        elif method == "regime_switch":
            prices, _ = self.regime_switch.generate(config, **kwargs)
            return prices
        else:
            raise ValueError(f"Unknown method: {method}")

    def generate_ohlcv(self, prices: np.ndarray, volume_mean: float = 1000.0,
                       volume_std: float = 500.0, seed: int | None = None) -> list[dict]:
        """Convert price path to OHLCV candlestick data."""
        rng = np.random.RandomState(seed)
        candles = []

        for i in range(0, len(prices) - 1):
            open_price = prices[i]
            close_price = prices[i + 1]
            high_mult = 1 + abs(rng.normal(0, 0.005))
            low_mult = 1 - abs(rng.normal(0, 0.005))

            high = max(open_price, close_price) * high_mult
            low = min(open_price, close_price) * low_mult
            volume = max(0, rng.normal(volume_mean, volume_std))

            candles.append({
                "open": float(open_price),
                "high": float(high),
                "low": float(low),
                "close": float(close_price),
                "volume": float(volume),
            })

        return candles

    def generate_training_dataset(self, n_samples: int = 1000,
                                  methods: list[str] | None = None,
                                  seed: int = 42) -> dict:
        """Generate a diverse training dataset using multiple methods."""
        if methods is None:
            methods = ["gbm", "jump_diffusion", "regime_switch"]

        rng = np.random.RandomState(seed)
        all_prices = []
        all_labels = []
        metadata = []

        samples_per_method = n_samples // len(methods)

        for method in methods:
            for i in range(samples_per_method):
                config = SyntheticConfig(
                    base_price=rng.uniform(1000, 100000),
                    n_steps=100,
                    seed=seed + i + methods.index(method) * samples_per_method,
                )

                try:
                    if method == "regime_switch":
                        prices, regimes = self.regime_switch.generate(config)
                    else:
                        prices = self.generate_price_path(method, config)

                    returns = np.diff(prices) / prices[:-1]
                    label = 1 if returns[-1] > 0 else 0  # Simple up/down label

                    all_prices.append(prices)
                    all_labels.append(label)
                    metadata.append({"method": method, "base_price": config.base_price})
                except Exception as e:
                    logger.warning("synthetic_generation_failed", method=method, error=str(e))

        logger.info("training_dataset_generated", n_samples=len(all_prices),
                    methods=methods, label_balance=sum(all_labels) / max(len(all_labels), 1))

        return {
            "prices": all_prices,
            "labels": all_labels,
            "metadata": metadata,
            "n_samples": len(all_prices),
        }

    def stress_test_portfolio(self, portfolio_value: float,
                              positions: dict[str, float],
                              scenarios: list[str] | None = None) -> list[dict]:
        """Run portfolio through stress test scenarios."""
        if scenarios is None:
            scenarios = list(HistoricalScenarioReplay.SCENARIOS.keys())

        results = []
        for scenario_name in scenarios:
            config = SyntheticConfig(base_price=portfolio_value, n_steps=200)
            try:
                prices, info = self.scenario_replay.replay(scenario_name, config)
                max_drawdown = float(np.max(1 - prices / np.maximum.accumulate(prices)))
                min_value = float(np.min(prices))

                results.append({
                    "scenario": scenario_name,
                    "description": info["description"],
                    "total_return_pct": info["total_return"] * 100,
                    "max_drawdown_pct": max_drawdown * 100,
                    "min_portfolio_value": min_value,
                    "recovery_days": len(prices) - int(np.argmin(prices)),
                    "worst_daily_return_pct": float(np.min(np.diff(prices) / prices[:-1]) * 100),
                    "phases": info["phases"],
                })
            except Exception as e:
                logger.warning("stress_test_failed", scenario=scenario_name, error=str(e))

        return results


# Module-level instance
synthetic_generator = SyntheticDataGenerator()
