"""Portfolio optimizer using Modern Portfolio Theory (Markowitz)."""
import numpy as np
from dataclasses import dataclass

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PortfolioAllocation:
    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float


class PortfolioOptimizer:
    """Markowitz mean-variance optimization."""

    def efficient_frontier(self, returns: np.ndarray, symbols: list[str], n_portfolios: int = 1000, risk_free_rate: float = 0.02) -> list[PortfolioAllocation]:
        n_assets = returns.shape[1]
        mean_returns = np.mean(returns, axis=0) * 252
        cov_matrix = np.cov(returns.T) * 252
        results = []

        for _ in range(n_portfolios):
            weights = np.random.dirichlet(np.ones(n_assets))
            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0

            results.append(PortfolioAllocation(
                weights=dict(zip(symbols, weights.tolist())),
                expected_return=float(port_return),
                volatility=float(port_vol),
                sharpe_ratio=float(sharpe),
            ))

        return sorted(results, key=lambda x: x.sharpe_ratio, reverse=True)

    def optimal_allocation(self, returns: np.ndarray, symbols: list[str], risk_free_rate: float = 0.02) -> PortfolioAllocation:
        frontier = self.efficient_frontier(returns, symbols, n_portfolios=5000, risk_free_rate=risk_free_rate)
        return frontier[0]

    def min_variance_portfolio(self, returns: np.ndarray, symbols: list[str]) -> PortfolioAllocation:
        frontier = self.efficient_frontier(returns, symbols, n_portfolios=5000)
        return min(frontier, key=lambda x: x.volatility)


portfolio_optimizer = PortfolioOptimizer()
