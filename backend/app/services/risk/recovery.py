"""Drawdown recovery estimation using historical simulation."""
import numpy as np
from dataclasses import dataclass

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RecoveryEstimate:
    estimated_days_median: int
    estimated_days_pessimistic: int  # 25th percentile
    estimated_days_optimistic: int   # 75th percentile
    probability_recovery_30d: float
    probability_recovery_90d: float
    current_drawdown_pct: float


class RecoveryEstimator:
    """Estimate time to recover from current drawdown."""

    def estimate(
        self,
        current_drawdown: float,
        historical_returns: np.ndarray,
        n_simulations: int = 5000,
    ) -> RecoveryEstimate:
        """Simulate recovery time from current drawdown level.

        Args:
            current_drawdown: Current drawdown as decimal (e.g., 0.10 = 10%)
            historical_returns: Array of historical daily returns
            n_simulations: Number of Monte Carlo paths
        """
        if current_drawdown <= 0 or len(historical_returns) < 10:
            return RecoveryEstimate(
                estimated_days_median=0,
                estimated_days_pessimistic=0,
                estimated_days_optimistic=0,
                probability_recovery_30d=1.0,
                probability_recovery_90d=1.0,
                current_drawdown_pct=round(current_drawdown * 100, 2),
            )

        recovery_needed = 1 / (1 - current_drawdown)  # e.g., 10% DD needs 11.1% gain
        recovery_days = []
        max_days = 365

        for _ in range(n_simulations):
            cumulative = 1.0
            for day in range(1, max_days + 1):
                daily_return = np.random.choice(historical_returns)
                cumulative *= (1 + daily_return)
                if cumulative >= recovery_needed:
                    recovery_days.append(day)
                    break
            else:
                recovery_days.append(max_days)

        recovery_arr = np.array(recovery_days)

        return RecoveryEstimate(
            estimated_days_median=int(np.median(recovery_arr)),
            estimated_days_pessimistic=int(np.percentile(recovery_arr, 75)),
            estimated_days_optimistic=int(np.percentile(recovery_arr, 25)),
            probability_recovery_30d=round(float(np.mean(recovery_arr <= 30)), 4),
            probability_recovery_90d=round(float(np.mean(recovery_arr <= 90)), 4),
            current_drawdown_pct=round(current_drawdown * 100, 2),
        )


recovery_estimator = RecoveryEstimator()
