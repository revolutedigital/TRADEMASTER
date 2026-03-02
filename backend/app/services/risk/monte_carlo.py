"""Monte Carlo simulation for portfolio risk analysis."""
import numpy as np
from dataclasses import dataclass

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MonteCarloResult:
    median_outcome: float
    worst_5pct: float
    best_5pct: float
    probability_of_loss: float
    expected_value: float
    paths_sample: list[list[float]]  # Sample paths for visualization


class MonteCarloSimulator:
    """Simulate future portfolio outcomes using historical return distributions."""

    def simulate(
        self,
        portfolio_value: float,
        returns: np.ndarray,
        n_simulations: int = 10000,
        horizon_days: int = 30,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation.

        Args:
            portfolio_value: Current portfolio value
            returns: Historical daily returns array
            n_simulations: Number of simulation paths
            horizon_days: Number of days to simulate forward
        """
        if len(returns) < 10:
            return MonteCarloResult(
                median_outcome=portfolio_value,
                worst_5pct=portfolio_value * 0.9,
                best_5pct=portfolio_value * 1.1,
                probability_of_loss=0.5,
                expected_value=portfolio_value,
                paths_sample=[],
            )

        simulated_paths = np.zeros((n_simulations, horizon_days))

        for i in range(n_simulations):
            random_returns = np.random.choice(returns, size=horizon_days, replace=True)
            simulated_paths[i] = portfolio_value * np.cumprod(1 + random_returns)

        final_values = simulated_paths[:, -1]

        result = MonteCarloResult(
            median_outcome=round(float(np.median(final_values)), 2),
            worst_5pct=round(float(np.percentile(final_values, 5)), 2),
            best_5pct=round(float(np.percentile(final_values, 95)), 2),
            probability_of_loss=round(float(np.mean(final_values < portfolio_value)), 4),
            expected_value=round(float(np.mean(final_values)), 2),
            paths_sample=simulated_paths[:100].tolist(),
        )

        logger.info(
            "monte_carlo_complete",
            simulations=n_simulations,
            horizon=horizon_days,
            median=result.median_outcome,
            prob_loss=result.probability_of_loss,
        )
        return result


monte_carlo_simulator = MonteCarloSimulator()
