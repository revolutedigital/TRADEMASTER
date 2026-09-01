"""Value-at-Risk (VaR) and Conditional VaR calculations."""

import numpy as np
from numpy.typing import NDArray
from typing import TypeAlias

from app.core.logging import get_logger

logger = get_logger(__name__)

FloatArray: TypeAlias = NDArray[np.float64]


class VaRCalculator:
    """Calculates Value-at-Risk metrics for portfolio risk assessment."""

    def historical_var(
        self,
        returns: list[float] | FloatArray,
        confidence: float = 0.95,
        portfolio_value: float = 1.0,
    ) -> float:
        """Historical VaR: based on actual return distribution.

        Returns the maximum expected loss at the given confidence level.
        """
        if len(returns) < 10:
            return 0.0

        arr: FloatArray = np.asarray(returns, dtype=np.float64)
        var_pct = self._historical_tail_threshold(arr, confidence)
        return abs(var_pct * portfolio_value)

    def parametric_var(
        self,
        portfolio_value: float,
        volatility: float,
        confidence: float = 0.95,
        holding_period_days: int = 1,
    ) -> float:
        """Parametric (Gaussian) VaR.

        Assumes normally distributed returns.
        """
        from scipy.stats import norm
        z_score = norm.ppf(1 - confidence)
        var = portfolio_value * volatility * abs(z_score) * np.sqrt(holding_period_days)
        return float(var)

    def cornish_fisher_var(
        self,
        returns: list[float] | FloatArray,
        confidence: float = 0.95,
        portfolio_value: float = 1.0,
    ) -> float:
        """VaR with Cornish-Fisher expansion for non-normal distributions.

        Adjusts the standard normal quantile using observed skewness and
        excess kurtosis, giving a more accurate tail-risk estimate when the
        return distribution departs from normality.
        """
        if len(returns) < 10:
            return 0.0

        from scipy.stats import norm, skew, kurtosis

        arr: FloatArray = np.asarray(returns, dtype=np.float64)
        z = norm.ppf(1 - confidence)
        s = float(skew(arr))
        k = float(kurtosis(arr, fisher=True))  # excess kurtosis

        # Cornish-Fisher expansion
        z_cf = (
            z
            + (z**2 - 1) * s / 6
            + (z**3 - 3 * z) * k / 24
            - (2 * z**3 - 5 * z) * s**2 / 36
        )

        mu = float(np.mean(arr))
        sigma = float(np.std(arr))
        var = -(mu + z_cf * sigma)
        return float(max(var, 0.0) * portfolio_value)

    def conditional_var(
        self,
        returns: list[float] | FloatArray,
        confidence: float = 0.95,
        portfolio_value: float = 1.0,
    ) -> float:
        """Conditional VaR (CVaR / Expected Shortfall).

        Average loss in the worst (1-confidence)% of scenarios.
        More conservative than VaR.
        """
        if len(returns) < 10:
            return 0.0

        arr: FloatArray = np.asarray(returns, dtype=np.float64)
        var_threshold = self._historical_tail_threshold(arr, confidence)
        tail_losses = arr[arr <= var_threshold]

        if len(tail_losses) == 0:
            return abs(float(var_threshold * portfolio_value))

        cvar = float(np.mean(tail_losses))
        return abs(cvar * portfolio_value)

    @staticmethod
    def _historical_tail_threshold(arr: FloatArray, confidence: float) -> float:
        """Return the empirical tail quantile without interpolation.

        A discrete sample must not manufacture an unobserved return between two
        observations. The selected index leaves at most ``1 - confidence`` of
        sample observations strictly below the VaR threshold.
        """
        sorted_returns = np.sort(arr)
        index = int(np.floor((1 - confidence) * len(sorted_returns)))
        index = max(0, min(index, len(sorted_returns) - 1))
        return float(sorted_returns[index])

    def calculate_all(
        self,
        returns: list[float] | FloatArray,
        portfolio_value: float,
        confidence: float = 0.95,
    ) -> dict[str, float | int]:
        """Calculate all VaR metrics at once."""
        arr: FloatArray = np.asarray(returns, dtype=np.float64)

        if len(arr) < 10:
            return {
                "historical_var": 0.0,
                "conditional_var": 0.0,
                "cornish_fisher_var": 0.0,
                "confidence": confidence,
                "portfolio_value": portfolio_value,
                "data_points": len(arr),
            }

        return {
            "historical_var": round(self.historical_var(arr, confidence, portfolio_value), 2),
            "conditional_var": round(self.conditional_var(arr, confidence, portfolio_value), 2),
            "cornish_fisher_var": round(self.cornish_fisher_var(arr, confidence, portfolio_value), 2),
            "confidence": confidence,
            "portfolio_value": portfolio_value,
            "data_points": len(arr),
            "max_daily_loss": round(abs(float(np.min(arr))) * portfolio_value, 2),
            "avg_daily_return": round(float(np.mean(arr)) * portfolio_value, 2),
            "volatility": round(float(np.std(arr)), 6),
        }


var_calculator = VaRCalculator()
