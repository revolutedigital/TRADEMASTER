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
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional VaR 95%
    paths_sample: list[list[float]]  # 100 sample paths for visualization


class MonteCarloSimulator:
    def simulate(
        self,
        portfolio_value: float,
        returns: list[float],
        n_simulations: int = 10000,
        horizon_days: int = 30,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation using historical returns distribution."""
        if not returns or len(returns) < 10:
            return MonteCarloResult(
                median_outcome=portfolio_value,
                worst_5pct=portfolio_value,
                best_5pct=portfolio_value,
                probability_of_loss=0.5,
                expected_value=portfolio_value,
                var_95=0,
                cvar_95=0,
                paths_sample=[],
            )

        ret = np.array(returns)
        simulated_paths = np.zeros((n_simulations, horizon_days))

        for i in range(n_simulations):
            random_returns = np.random.choice(ret, size=horizon_days, replace=True)
            simulated_paths[i] = portfolio_value * np.cumprod(1 + random_returns)

        final_values = simulated_paths[:, -1]

        # VaR and CVaR
        sorted_losses = np.sort(portfolio_value - final_values)
        var_95_idx = int(0.95 * len(sorted_losses))
        var_95 = float(sorted_losses[var_95_idx])
        cvar_95 = float(np.mean(sorted_losses[var_95_idx:]))

        # Sample 100 paths for visualization
        sample_indices = np.random.choice(n_simulations, min(100, n_simulations), replace=False)
        paths_sample = simulated_paths[sample_indices].tolist()

        result = MonteCarloResult(
            median_outcome=float(np.median(final_values)),
            worst_5pct=float(np.percentile(final_values, 5)),
            best_5pct=float(np.percentile(final_values, 95)),
            probability_of_loss=float(np.mean(final_values < portfolio_value)),
            expected_value=float(np.mean(final_values)),
            var_95=var_95,
            cvar_95=cvar_95,
            paths_sample=paths_sample,
        )

        logger.info(
            "monte_carlo_complete",
            portfolio=portfolio_value,
            horizon=horizon_days,
            median=round(result.median_outcome, 2),
            var_95=round(result.var_95, 2),
            prob_loss=round(result.probability_of_loss, 3),
        )
        return result


monte_carlo = MonteCarloSimulator()
