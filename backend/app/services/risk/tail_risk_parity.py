"""Tail risk parity portfolio construction - allocation based on CVaR contributions."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy.stats import norm

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AllocationResult:
    """Portfolio allocation output with risk diagnostics."""
    weights: dict[str, float]
    method: str
    cvar_contributions: dict[str, float]
    max_deviation_pct: float
    needs_rebalance: bool
    timestamp: str


class TailRiskParityPortfolio:
    """
    Portfolio optimizer that equalizes tail-risk (CVaR) contributions.

    Instead of classic risk parity (which targets equal volatility
    contribution), this allocator ensures each asset contributes
    equally to the portfolio's Expected Shortfall.  This gives more
    protection against fat-tailed events common in crypto markets.

    Features:
    - CVaR-based allocation using historical return samples
    - Cornish-Fisher expansion to incorporate skewness and kurtosis
    - Expected Shortfall parity via iterative re-weighting
    - Benchmark comparison with equal-weight and vol-parity
    - Automatic rebalance trigger when any asset's tail risk
      contribution deviates more than 10% from target
    """

    DEFAULT_CONFIDENCE = 0.95
    REBALANCE_THRESHOLD = 0.10  # 10% relative deviation triggers rebalance
    MAX_ITERATIONS = 200
    CONVERGENCE_TOL = 1e-6

    def __init__(
        self,
        confidence: float = 0.95,
        rebalance_threshold: float = 0.10,
    ):
        self.confidence = confidence
        self.rebalance_threshold = rebalance_threshold

        self._returns: dict[str, list[float]] = {}
        self._current_weights: dict[str, float] = {}

        logger.info(
            "tail_risk_parity_initialized",
            confidence=confidence,
            rebalance_threshold=rebalance_threshold,
        )

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def update_returns(self, symbol: str, returns: list[float]) -> None:
        """Set the return series for an asset.

        Args:
            symbol: Asset identifier.
            returns: Historical returns (fractional, e.g. 0.02 for +2%).
        """
        self._returns[symbol] = list(returns)

    def add_return(self, symbol: str, ret: float) -> None:
        """Append a single return observation."""
        self._returns.setdefault(symbol, []).append(ret)
        # Keep bounded
        if len(self._returns[symbol]) > 1000:
            self._returns[symbol] = self._returns[symbol][-1000:]

    # ------------------------------------------------------------------
    # CVaR / Expected Shortfall
    # ------------------------------------------------------------------

    def historical_cvar(
        self,
        returns: np.ndarray,
        confidence: float | None = None,
    ) -> float:
        """Historical Conditional VaR (Expected Shortfall).

        The average of losses beyond the VaR threshold.

        Args:
            returns: Array of return observations.
            confidence: Confidence level (default: instance level).

        Returns:
            CVaR as a positive number representing expected tail loss.
        """
        if len(returns) < 10:
            return 0.0

        conf = confidence or self.confidence
        cutoff = np.percentile(returns, (1 - conf) * 100)
        tail = returns[returns <= cutoff]
        if len(tail) == 0:
            return abs(float(cutoff))
        return abs(float(np.mean(tail)))

    def cornish_fisher_var(
        self,
        returns: np.ndarray,
        confidence: float | None = None,
    ) -> float:
        """VaR using the Cornish-Fisher expansion.

        Adjusts the Gaussian quantile for observed skewness (S) and
        excess kurtosis (K):

            z_cf = z + (z^2 - 1)*S/6 + (z^3 - 3*z)*K/24 - (2*z^3 - 5*z)*S^2/36

        This captures non-normality without requiring a full
        distributional fit.

        Args:
            returns: Array of return observations.
            confidence: Confidence level.

        Returns:
            Cornish-Fisher adjusted VaR as a positive loss.
        """
        if len(returns) < 20:
            return 0.0

        conf = confidence or self.confidence
        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))
        if sigma < 1e-12:
            return 0.0

        z = norm.ppf(1 - conf)  # negative for high confidence
        standardized = (returns - mu) / sigma

        skew = float(np.mean(standardized ** 3))
        excess_kurt = float(np.mean(standardized ** 4)) - 3.0

        z_cf = (
            z
            + (z ** 2 - 1) * skew / 6
            + (z ** 3 - 3 * z) * excess_kurt / 24
            - (2 * z ** 3 - 5 * z) * skew ** 2 / 36
        )

        cf_var = -(mu + z_cf * sigma)
        return max(0.0, round(float(cf_var), 8))

    def cornish_fisher_cvar(
        self,
        returns: np.ndarray,
        confidence: float | None = None,
    ) -> float:
        """CVaR estimated via Cornish-Fisher expansion.

        Uses the CF-VaR as the threshold and averages the losses beyond
        that point from the empirical distribution.  Falls back to
        historical CVaR when the CF threshold yields no tail samples.

        Args:
            returns: Array of return observations.
            confidence: Confidence level.

        Returns:
            Cornish-Fisher adjusted CVaR as a positive loss.
        """
        if len(returns) < 20:
            return self.historical_cvar(returns, confidence)

        cf_threshold = -self.cornish_fisher_var(returns, confidence)
        tail = returns[returns <= cf_threshold]

        if len(tail) == 0:
            return self.historical_cvar(returns, confidence)

        return abs(float(np.mean(tail)))

    # ------------------------------------------------------------------
    # Portfolio CVaR contribution
    # ------------------------------------------------------------------

    def _portfolio_returns(
        self, weights: np.ndarray, returns_matrix: np.ndarray
    ) -> np.ndarray:
        """Compute portfolio return series from asset returns and weights."""
        return returns_matrix @ weights

    def _cvar_contributions(
        self,
        weights: np.ndarray,
        returns_matrix: np.ndarray,
        confidence: float | None = None,
    ) -> np.ndarray:
        """Estimate each asset's marginal contribution to portfolio CVaR.

        Uses the Euler decomposition: the contribution of asset i
        equals w_i * d(CVaR)/d(w_i), estimated numerically.

        Args:
            weights: Current weight vector.
            returns_matrix: (T x N) matrix of asset returns.
            confidence: Confidence level.

        Returns:
            Array of CVaR contributions (one per asset).
        """
        conf = confidence or self.confidence
        n_assets = len(weights)
        port_ret = self._portfolio_returns(weights, returns_matrix)
        cutoff = np.percentile(port_ret, (1 - conf) * 100)
        tail_mask = port_ret <= cutoff

        if tail_mask.sum() == 0:
            return np.ones(n_assets) / n_assets

        # Marginal contribution: E[r_i | portfolio in tail]
        tail_returns = returns_matrix[tail_mask]
        marginal = np.mean(tail_returns, axis=0)

        # Contribution = w_i * marginal_i
        contributions = weights * marginal
        # Normalize to sum to total portfolio CVaR (with sign)
        total = np.sum(contributions)
        if abs(total) < 1e-12:
            return np.ones(n_assets) / n_assets

        # Return absolute contributions (positive = contributes to risk)
        return -contributions  # negate because tail returns are negative

    # ------------------------------------------------------------------
    # Allocation methods
    # ------------------------------------------------------------------

    def tail_risk_parity_weights(
        self,
        symbols: list[str] | None = None,
        use_cornish_fisher: bool = True,
    ) -> AllocationResult:
        """Compute tail-risk-parity weights via iterative re-weighting.

        Each asset should contribute equally to the portfolio's CVaR.
        The algorithm iteratively adjusts weights inversely proportional
        to each asset's marginal CVaR contribution until convergence.

        Args:
            symbols: Subset of assets to include. Defaults to all with data.
            use_cornish_fisher: If True, use Cornish-Fisher CVaR.

        Returns:
            AllocationResult with optimized weights.
        """
        if symbols is None:
            symbols = [s for s in self._returns if len(self._returns[s]) >= 20]

        if len(symbols) < 2:
            logger.warning("tail_risk_parity_insufficient_assets", n=len(symbols))
            w = {s: 1.0 for s in symbols} if symbols else {}
            return AllocationResult(
                weights=w,
                method="tail_risk_parity",
                cvar_contributions={s: 1.0 for s in symbols} if symbols else {},
                max_deviation_pct=0.0,
                needs_rebalance=False,
                timestamp=datetime.utcnow().isoformat(),
            )

        n = len(symbols)
        # Build aligned returns matrix (use shortest common length)
        min_len = min(len(self._returns[s]) for s in symbols)
        returns_matrix = np.column_stack(
            [np.array(self._returns[s][-min_len:]) for s in symbols]
        )

        # Compute standalone CVaR for each asset
        standalone_cvar = np.array([
            (self.cornish_fisher_cvar(returns_matrix[:, i])
             if use_cornish_fisher
             else self.historical_cvar(returns_matrix[:, i]))
            for i in range(n)
        ])

        # Initialize with inverse-CVaR weights
        safe_cvar = np.where(standalone_cvar > 1e-12, standalone_cvar, 1e-12)
        weights = (1.0 / safe_cvar)
        weights /= weights.sum()

        # Iterative refinement
        for iteration in range(self.MAX_ITERATIONS):
            contributions = self._cvar_contributions(weights, returns_matrix)
            total_contrib = contributions.sum()
            if total_contrib < 1e-12:
                break

            # Target: equal fraction
            target_each = total_contrib / n
            ratios = contributions / target_each

            # Adjust weights inversely to over/under-contribution
            safe_ratios = np.where(ratios > 1e-8, ratios, 1e-8)
            new_weights = weights / safe_ratios
            new_weights = np.maximum(new_weights, 1e-8)
            new_weights /= new_weights.sum()

            delta = np.max(np.abs(new_weights - weights))
            weights = new_weights

            if delta < self.CONVERGENCE_TOL:
                logger.debug("tail_risk_parity_converged", iterations=iteration + 1)
                break

        # Final contribution calculation
        contributions = self._cvar_contributions(weights, returns_matrix)
        total_contrib = contributions.sum()
        if total_contrib > 1e-12:
            pct_contributions = contributions / total_contrib
        else:
            pct_contributions = np.ones(n) / n

        target_pct = 1.0 / n
        deviations = np.abs(pct_contributions - target_pct) / target_pct
        max_dev = float(np.max(deviations))

        weight_dict = {symbols[i]: round(float(weights[i]), 6) for i in range(n)}
        contrib_dict = {symbols[i]: round(float(pct_contributions[i]), 6) for i in range(n)}

        self._current_weights = weight_dict

        result = AllocationResult(
            weights=weight_dict,
            method="tail_risk_parity",
            cvar_contributions=contrib_dict,
            max_deviation_pct=round(max_dev * 100, 2),
            needs_rebalance=False,  # freshly optimized
            timestamp=datetime.utcnow().isoformat(),
        )

        logger.info(
            "tail_risk_parity_computed",
            n_assets=n,
            max_deviation_pct=result.max_deviation_pct,
            weights=weight_dict,
        )

        return result

    def equal_weight(self, symbols: list[str] | None = None) -> AllocationResult:
        """Equal-weight benchmark allocation.

        Args:
            symbols: Asset list. Defaults to all with data.

        Returns:
            AllocationResult with uniform weights.
        """
        if symbols is None:
            symbols = list(self._returns.keys())

        n = len(symbols)
        if n == 0:
            return AllocationResult(
                weights={}, method="equal_weight",
                cvar_contributions={}, max_deviation_pct=0.0,
                needs_rebalance=False, timestamp=datetime.utcnow().isoformat(),
            )

        w = round(1.0 / n, 6)
        weights = {s: w for s in symbols}

        # Compute CVaR contributions at equal weights
        contrib_dict = self._compute_contribution_dict(symbols, weights)

        return AllocationResult(
            weights=weights,
            method="equal_weight",
            cvar_contributions=contrib_dict,
            max_deviation_pct=self._max_deviation(contrib_dict),
            needs_rebalance=False,
            timestamp=datetime.utcnow().isoformat(),
        )

    def volatility_parity(self, symbols: list[str] | None = None) -> AllocationResult:
        """Volatility (inverse-vol) parity benchmark.

        Weights are proportional to 1/sigma, so that each asset
        contributes roughly equal volatility to the portfolio
        (assuming zero correlation, which is an approximation).

        Args:
            symbols: Asset list. Defaults to all with data.

        Returns:
            AllocationResult with inverse-volatility weights.
        """
        if symbols is None:
            symbols = [s for s in self._returns if len(self._returns[s]) >= 10]

        n = len(symbols)
        if n == 0:
            return AllocationResult(
                weights={}, method="volatility_parity",
                cvar_contributions={}, max_deviation_pct=0.0,
                needs_rebalance=False, timestamp=datetime.utcnow().isoformat(),
            )

        vols = np.array([
            np.std(self._returns[s][-252:], ddof=1)
            for s in symbols
        ])

        safe_vols = np.where(vols > 1e-12, vols, 1e-12)
        raw_weights = 1.0 / safe_vols
        raw_weights /= raw_weights.sum()

        weights = {symbols[i]: round(float(raw_weights[i]), 6) for i in range(n)}

        contrib_dict = self._compute_contribution_dict(symbols, weights)

        return AllocationResult(
            weights=weights,
            method="volatility_parity",
            cvar_contributions=contrib_dict,
            max_deviation_pct=self._max_deviation(contrib_dict),
            needs_rebalance=False,
            timestamp=datetime.utcnow().isoformat(),
        )

    def compare_allocations(
        self, symbols: list[str] | None = None
    ) -> dict:
        """Run all three allocation methods and compare.

        Args:
            symbols: Asset list.

        Returns:
            Dictionary comparing weights and risk contributions.
        """
        trp = self.tail_risk_parity_weights(symbols)
        ew = self.equal_weight(symbols)
        vp = self.volatility_parity(symbols)

        return {
            "tail_risk_parity": {
                "weights": trp.weights,
                "cvar_contributions": trp.cvar_contributions,
                "max_deviation_pct": trp.max_deviation_pct,
            },
            "equal_weight": {
                "weights": ew.weights,
                "cvar_contributions": ew.cvar_contributions,
                "max_deviation_pct": ew.max_deviation_pct,
            },
            "volatility_parity": {
                "weights": vp.weights,
                "cvar_contributions": vp.cvar_contributions,
                "max_deviation_pct": vp.max_deviation_pct,
            },
            "recommendation": self._pick_recommendation(trp, ew, vp),
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    def check_rebalance(
        self,
        symbols: list[str] | None = None,
    ) -> AllocationResult | None:
        """Check if the current weights need rebalancing.

        Re-evaluates CVaR contributions of the current weights.  If any
        asset's contribution deviates more than 10% (relative) from the
        equal-contribution target, a new allocation is computed.

        Returns:
            A new AllocationResult if rebalance is needed, else None.
        """
        if not self._current_weights:
            logger.info("check_rebalance_no_weights")
            return None

        active_symbols = symbols or list(self._current_weights.keys())
        valid = [s for s in active_symbols if s in self._returns and len(self._returns[s]) >= 20]

        if len(valid) < 2:
            return None

        contrib_dict = self._compute_contribution_dict(valid, self._current_weights)
        max_dev = self._max_deviation(contrib_dict)

        if max_dev > self.rebalance_threshold * 100:
            logger.info(
                "rebalance_triggered",
                max_deviation_pct=max_dev,
                threshold_pct=self.rebalance_threshold * 100,
            )
            return self.tail_risk_parity_weights(valid)

        logger.debug("rebalance_not_needed", max_deviation_pct=max_dev)
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_contribution_dict(
        self,
        symbols: list[str],
        weights: dict[str, float],
    ) -> dict[str, float]:
        """Compute CVaR contribution percentages for given weights."""
        valid = [s for s in symbols if s in self._returns and len(self._returns[s]) >= 10]
        if len(valid) < 2:
            return {s: round(1.0 / max(len(valid), 1), 6) for s in valid}

        min_len = min(len(self._returns[s]) for s in valid)
        returns_matrix = np.column_stack(
            [np.array(self._returns[s][-min_len:]) for s in valid]
        )

        w_arr = np.array([weights.get(s, 0.0) for s in valid])
        total_w = w_arr.sum()
        if total_w > 0:
            w_arr /= total_w

        contributions = self._cvar_contributions(w_arr, returns_matrix)
        total = contributions.sum()

        if total > 1e-12:
            pct = contributions / total
        else:
            pct = np.ones(len(valid)) / len(valid)

        return {valid[i]: round(float(pct[i]), 6) for i in range(len(valid))}

    @staticmethod
    def _max_deviation(contrib_dict: dict[str, float]) -> float:
        """Max relative deviation of any contribution from equal share."""
        n = len(contrib_dict)
        if n == 0:
            return 0.0
        target = 1.0 / n
        if target < 1e-12:
            return 0.0
        deviations = [abs(v - target) / target for v in contrib_dict.values()]
        return round(max(deviations) * 100, 2) if deviations else 0.0

    @staticmethod
    def _pick_recommendation(
        trp: AllocationResult,
        ew: AllocationResult,
        vp: AllocationResult,
    ) -> str:
        """Pick a recommendation based on allocation comparison."""
        if trp.max_deviation_pct <= 5.0:
            return (
                "Tail risk parity achieved near-perfect balance. "
                "Recommended for downside protection."
            )
        if vp.max_deviation_pct < trp.max_deviation_pct:
            return (
                "Volatility parity shows better balance than tail risk parity; "
                "return distributions may be close to Gaussian. "
                "Consider vol-parity for lower computational cost."
            )
        return (
            "Tail risk parity provides the best tail-risk equalization. "
            "Use it for portfolios with significant skew or kurtosis."
        )


# Module-level instance
tail_risk_parity = TailRiskParityPortfolio()
