"""Monte Carlo validation: statistical significance testing for backtest results.

Shuffles the order of trades N times to create a distribution of outcomes.
If the original strategy's performance is in the top percentiles of random
orderings, the result is statistically significant (not just luck).

Key outputs:
- p_value: probability that random ordering beats the strategy
- confidence_intervals: 90%, 95%, 99% CI for key metrics
- worst_case / best_case from simulations
"""

from dataclasses import dataclass

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SIMULATIONS = 1000


@dataclass
class MonteCarloResult:
    """Result of Monte Carlo significance testing."""

    original_return_pct: float
    original_sharpe: float
    original_max_dd_pct: float

    # Statistical significance
    p_value_return: float    # P(random >= original return)
    p_value_sharpe: float    # P(random >= original sharpe)
    is_significant_95: bool  # p_value < 0.05
    is_significant_99: bool  # p_value < 0.01

    # Distribution stats
    sim_mean_return: float
    sim_std_return: float
    sim_median_return: float

    # Confidence intervals for return
    ci_90: tuple[float, float]  # 5th-95th percentile
    ci_95: tuple[float, float]  # 2.5th-97.5th percentile
    ci_99: tuple[float, float]  # 0.5th-99.5th percentile

    # Worst/best case
    worst_return_pct: float  # 1st percentile
    best_return_pct: float   # 99th percentile
    worst_max_dd_pct: float  # 99th percentile of drawdowns

    # Simulation details
    n_simulations: int
    n_trades: int


def run_monte_carlo(
    trade_returns: list[float],
    initial_capital: float = 10000.0,
    n_simulations: int = DEFAULT_SIMULATIONS,
) -> MonteCarloResult:
    """Run Monte Carlo simulation by shuffling trade order.

    Args:
        trade_returns: List of individual trade return percentages (e.g. [0.02, -0.01, 0.03])
        initial_capital: Starting capital for equity curve simulation
        n_simulations: Number of random shuffles

    Returns:
        MonteCarloResult with statistical significance metrics
    """
    returns = np.array(trade_returns, dtype=float)
    n_trades = len(returns)

    if n_trades < 5:
        logger.warning("monte_carlo_insufficient_trades", n_trades=n_trades)
        return _empty_result(n_trades)

    # Original strategy metrics
    orig_equity = _simulate_equity(returns, initial_capital)
    orig_return = (orig_equity[-1] - initial_capital) / initial_capital
    orig_sharpe = _compute_sharpe(returns)
    orig_max_dd = _max_drawdown(orig_equity)

    # Run simulations
    sim_returns = np.zeros(n_simulations)
    sim_sharpes = np.zeros(n_simulations)
    sim_max_dds = np.zeros(n_simulations)

    rng = np.random.default_rng(seed=42)

    for i in range(n_simulations):
        shuffled = rng.permutation(returns)
        equity = _simulate_equity(shuffled, initial_capital)
        sim_returns[i] = (equity[-1] - initial_capital) / initial_capital
        sim_sharpes[i] = _compute_sharpe(shuffled)
        sim_max_dds[i] = _max_drawdown(equity)

    # P-values: what fraction of random orderings beat the original?
    p_value_return = float(np.mean(sim_returns >= orig_return))
    p_value_sharpe = float(np.mean(sim_sharpes >= orig_sharpe))

    # Confidence intervals
    ci_90 = (float(np.percentile(sim_returns, 5)), float(np.percentile(sim_returns, 95)))
    ci_95 = (float(np.percentile(sim_returns, 2.5)), float(np.percentile(sim_returns, 97.5)))
    ci_99 = (float(np.percentile(sim_returns, 0.5)), float(np.percentile(sim_returns, 99.5)))

    result = MonteCarloResult(
        original_return_pct=round(orig_return * 100, 4),
        original_sharpe=round(orig_sharpe, 4),
        original_max_dd_pct=round(orig_max_dd * 100, 4),
        p_value_return=round(p_value_return, 4),
        p_value_sharpe=round(p_value_sharpe, 4),
        is_significant_95=p_value_return < 0.05,
        is_significant_99=p_value_return < 0.01,
        sim_mean_return=round(float(np.mean(sim_returns)) * 100, 4),
        sim_std_return=round(float(np.std(sim_returns)) * 100, 4),
        sim_median_return=round(float(np.median(sim_returns)) * 100, 4),
        ci_90=(round(ci_90[0] * 100, 4), round(ci_90[1] * 100, 4)),
        ci_95=(round(ci_95[0] * 100, 4), round(ci_95[1] * 100, 4)),
        ci_99=(round(ci_99[0] * 100, 4), round(ci_99[1] * 100, 4)),
        worst_return_pct=round(float(np.percentile(sim_returns, 1)) * 100, 4),
        best_return_pct=round(float(np.percentile(sim_returns, 99)) * 100, 4),
        worst_max_dd_pct=round(float(np.percentile(sim_max_dds, 99)) * 100, 4),
        n_simulations=n_simulations,
        n_trades=n_trades,
    )

    logger.info(
        "monte_carlo_complete",
        n_trades=n_trades,
        n_simulations=n_simulations,
        orig_return=result.original_return_pct,
        p_value=result.p_value_return,
        significant_95=result.is_significant_95,
    )

    return result


def _simulate_equity(returns: np.ndarray, initial: float) -> np.ndarray:
    """Build equity curve from trade returns."""
    equity = np.zeros(len(returns) + 1)
    equity[0] = initial
    for i, r in enumerate(returns):
        equity[i + 1] = equity[i] * (1 + r)
    return equity


def _compute_sharpe(returns: np.ndarray, annualize_factor: float = 252.0) -> float:
    """Compute Sharpe ratio from trade returns."""
    if len(returns) < 2:
        return 0.0
    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns))
    if std_r < 1e-10:
        return 0.0
    return mean_r / std_r * np.sqrt(annualize_factor)


def _max_drawdown(equity: np.ndarray) -> float:
    """Compute maximum drawdown from equity curve."""
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / np.where(peak > 0, peak, 1)
    return float(np.min(drawdown))


def _empty_result(n_trades: int) -> MonteCarloResult:
    return MonteCarloResult(
        original_return_pct=0, original_sharpe=0, original_max_dd_pct=0,
        p_value_return=1.0, p_value_sharpe=1.0,
        is_significant_95=False, is_significant_99=False,
        sim_mean_return=0, sim_std_return=0, sim_median_return=0,
        ci_90=(0, 0), ci_95=(0, 0), ci_99=(0, 0),
        worst_return_pct=0, best_return_pct=0, worst_max_dd_pct=0,
        n_simulations=0, n_trades=n_trades,
    )
