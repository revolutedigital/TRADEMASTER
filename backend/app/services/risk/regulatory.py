"""Regulatory-inspired risk metrics (Basel III adapted for crypto)."""
from dataclasses import dataclass

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RegulatoryMetrics:
    capital_adequacy_ratio: float  # Total capital / Risk-weighted assets
    leverage_ratio: float          # Equity / Total exposure
    liquidity_coverage: float      # Available cash / 30-day outflow estimate
    risk_weighted_exposure: float
    tier1_capital: float


class RegulatoryCalculator:
    """Calculate Basel III-inspired risk metrics for crypto portfolios."""

    def calculate(
        self,
        total_equity: float,
        available_cash: float,
        total_exposure: float,
        unrealized_pnl: float,
        daily_volume: float = 0,
    ) -> RegulatoryMetrics:
        """Calculate regulatory risk metrics.

        Adapts traditional banking metrics for crypto:
        - Higher risk weights due to crypto volatility
        - 24/7 market means different liquidity assumptions
        """
        # Risk-weighted exposure (crypto gets 150% risk weight vs 100% for equities)
        risk_weight = 1.5
        rwa = total_exposure * risk_weight

        # Capital Adequacy Ratio (CAR) - min 8% in Basel III
        car = total_equity / rwa if rwa > 0 else float("inf")

        # Leverage ratio
        leverage = total_equity / total_exposure if total_exposure > 0 else float("inf")

        # Liquidity Coverage Ratio
        # Estimate 30-day outflow as 10% of exposure (conservative)
        estimated_outflow = total_exposure * 0.10
        lcr = available_cash / estimated_outflow if estimated_outflow > 0 else float("inf")

        # Tier 1 capital = equity - unrealized losses
        tier1 = max(0, total_equity + min(0, unrealized_pnl))

        result = RegulatoryMetrics(
            capital_adequacy_ratio=round(min(car, 99.99), 4),
            leverage_ratio=round(min(leverage, 99.99), 4),
            liquidity_coverage=round(min(lcr, 99.99), 4),
            risk_weighted_exposure=round(rwa, 2),
            tier1_capital=round(tier1, 2),
        )

        logger.info(
            "regulatory_metrics",
            car=result.capital_adequacy_ratio,
            leverage=result.leverage_ratio,
            lcr=result.liquidity_coverage,
        )
        return result


regulatory_calculator = RegulatoryCalculator()
