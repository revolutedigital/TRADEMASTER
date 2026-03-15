"""Monte Carlo simulation for portfolio risk analysis.

Supports two simulation methods:
- "bootstrap": classic resampling from historical returns (default)
- "t_distribution": fat-tailed Student's t simulation with EWMA volatility
  clustering, better suited for crypto markets where tail events are frequent
"""

import numpy as np
from dataclasses import dataclass
from scipy import stats as sp_stats

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
    method: str = "bootstrap"  # simulation method used


class MonteCarloSimulator:
    def simulate(
        self,
        portfolio_value: float,
        returns: list[float],
        n_simulations: int = 10000,
        horizon_days: int = 30,
        method: str = "bootstrap",
        ewma_lambda: float = 0.94,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation.

        Args:
            portfolio_value: Current portfolio value in USD.
            returns: Historical daily returns (e.g. [0.01, -0.005, ...]).
            n_simulations: Number of simulation paths.
            horizon_days: Projection horizon in days.
            method: "bootstrap" for classic resampling, "t_distribution" for
                    fat-tailed Student-t with EWMA volatility clustering.
            ewma_lambda: Decay factor for EWMA volatility (only used when
                         method="t_distribution"). 0.94 is the RiskMetrics
                         default; lower values react faster to recent vol.
        """
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
                method=method,
            )

        ret = np.array(returns, dtype=np.float64)

        if method == "t_distribution":
            simulated_paths = self._simulate_t_distribution(
                portfolio_value, ret, n_simulations, horizon_days, ewma_lambda,
            )
        else:
            simulated_paths = self._simulate_bootstrap(
                portfolio_value, ret, n_simulations, horizon_days,
            )

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
            method=method,
        )

        logger.info(
            "monte_carlo_complete",
            portfolio=portfolio_value,
            horizon=horizon_days,
            method=method,
            median=round(result.median_outcome, 2),
            var_95=round(result.var_95, 2),
            prob_loss=round(result.probability_of_loss, 3),
        )
        return result

    # ------------------------------------------------------------------
    # Private simulation engines
    # ------------------------------------------------------------------

    @staticmethod
    def _simulate_bootstrap(
        portfolio_value: float,
        ret: np.ndarray,
        n_simulations: int,
        horizon_days: int,
    ) -> np.ndarray:
        """Classic bootstrap resampling from historical returns."""
        simulated_paths = np.zeros((n_simulations, horizon_days))
        for i in range(n_simulations):
            random_returns = np.random.choice(ret, size=horizon_days, replace=True)
            simulated_paths[i] = portfolio_value * np.cumprod(1 + random_returns)
        return simulated_paths

    @staticmethod
    def _simulate_t_distribution(
        portfolio_value: float,
        ret: np.ndarray,
        n_simulations: int,
        horizon_days: int,
        ewma_lambda: float,
    ) -> np.ndarray:
        """Fat-tailed simulation: Student-t innovations + EWMA volatility.

        1. Fit degrees-of-freedom from historical excess kurtosis so the
           t-distribution matches observed tail weight.
        2. Use EWMA to estimate current conditional volatility from the
           historical series.
        3. At each simulation step, draw a t-distributed shock, scale it
           by the evolving EWMA volatility estimate.

        This addresses the audit finding that plain bootstrap ignores both
        fat tails and volatility clustering, both of which are dominant
        features of crypto return series.
        """
        mu = float(np.mean(ret))
        sigma = float(np.std(ret, ddof=1))

        # --- Fit Student-t degrees of freedom from kurtosis ---------------
        # Excess kurtosis of t(df) = 6 / (df - 4) for df > 4
        # Solving: df = 4 + 6 / kurtosis_excess (clamped to [3, 120])
        excess_kurt = float(sp_stats.kurtosis(ret, fisher=True))
        if excess_kurt > 0.05:
            df = 4.0 + 6.0 / excess_kurt
        else:
            # Near-normal tails; use high df so t converges to normal
            df = 120.0
        df = float(np.clip(df, 3.0, 120.0))

        # --- Seed EWMA variance from the historical series ----------------
        # Walk backwards through historical returns to build the EWMA
        # variance estimate at t=0 (the last observed return).
        lam = ewma_lambda
        var_ewma = sigma ** 2  # start from unconditional variance
        for r in ret:
            var_ewma = lam * var_ewma + (1.0 - lam) * (r - mu) ** 2

        # --- Simulate paths -----------------------------------------------
        simulated_paths = np.zeros((n_simulations, horizon_days))
        rng = np.random.default_rng()

        for i in range(n_simulations):
            vol2 = var_ewma  # each path starts from current EWMA vol
            cumulative = portfolio_value
            for t in range(horizon_days):
                vol = np.sqrt(max(vol2, 1e-20))
                # Draw from standardised t, then scale
                shock = rng.standard_t(df)
                daily_return = mu + vol * shock
                cumulative *= (1.0 + daily_return)
                simulated_paths[i, t] = cumulative
                # Update EWMA variance for next step
                vol2 = lam * vol2 + (1.0 - lam) * (daily_return - mu) ** 2

        return simulated_paths


monte_carlo = MonteCarloSimulator()
