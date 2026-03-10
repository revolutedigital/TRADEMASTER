"""Causal inference engine for discovering and quantifying causal relationships between assets."""
import numpy as np
from dataclasses import dataclass, field
from scipy import stats

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GrangerResult:
    """Result of a Granger causality test between two time series."""
    cause: str
    effect: str
    f_statistic: float
    p_value: float
    optimal_lag: int
    is_causal: bool  # True if p_value < significance level


@dataclass
class CausalEdge:
    """A directed edge in the causal graph."""
    parent: str
    child: str
    strength: float  # Standardized causal effect magnitude
    p_value: float
    lag: int  # Temporal lag in periods


@dataclass
class CausalGraph:
    """Structural causal model represented as a directed graph."""
    nodes: list[str]
    edges: list[CausalEdge]
    adjacency: dict[str, list[str]] = field(default_factory=dict)

    def parents(self, node: str) -> list[str]:
        """Return direct parents of a node."""
        return [e.parent for e in self.edges if e.child == node]

    def children(self, node: str) -> list[str]:
        """Return direct children of a node."""
        return [e.child for e in self.edges if e.parent == node]

    def ancestors(self, node: str, visited: set[str] | None = None) -> set[str]:
        """Return all ancestors of a node via recursive traversal."""
        visited = visited or set()
        for parent in self.parents(node):
            if parent not in visited:
                visited.add(parent)
                self.ancestors(parent, visited)
        return visited

    def descendants(self, node: str, visited: set[str] | None = None) -> set[str]:
        """Return all descendants of a node via recursive traversal."""
        visited = visited or set()
        for child in self.children(node):
            if child not in visited:
                visited.add(child)
                self.descendants(child, visited)
        return visited


@dataclass
class InterventionResult:
    """Result of a causal intervention analysis (do-calculus)."""
    treatment: str
    outcome: str
    treatment_change: float  # e.g., -0.05 for a 5% drop
    estimated_effect: float  # Estimated causal effect on outcome
    confidence_interval: tuple[float, float]
    adjustment_set: list[str]  # Variables adjusted for (backdoor/frontdoor)
    method: str  # "backdoor", "frontdoor", or "direct"


class CausalInferenceEngine:
    """Discover and quantify causal relationships between financial assets.

    Implements:
    - Granger causality testing for temporal causal discovery
    - Structural causal model (SCM) construction from pairwise tests
    - do-calculus-inspired intervention analysis
    - Backdoor and frontdoor adjustment for unbiased causal estimation
    """

    def __init__(self, significance: float = 0.05, max_lag: int = 10):
        self.significance = significance
        self.max_lag = max_lag

    # ------------------------------------------------------------------
    # Granger causality
    # ------------------------------------------------------------------

    def granger_test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        max_lag: int | None = None,
    ) -> GrangerResult:
        """Test whether *x* Granger-causes *y*.

        Compares a restricted model (y ~ own lags) against an unrestricted
        model (y ~ own lags + x lags) using an F-test.

        Args:
            x: Potential cause series (1-D array).
            y: Potential effect series (1-D array).
            max_lag: Maximum lag to consider. Uses instance default when None.

        Returns:
            GrangerResult with the best (lowest p-value) lag.
        """
        max_lag = max_lag or self.max_lag
        n = min(len(x), len(y))
        if n < max_lag + 2:
            logger.warning("Series too short (%d) for max_lag=%d", n, max_lag)
            return GrangerResult(
                cause="x", effect="y",
                f_statistic=0.0, p_value=1.0,
                optimal_lag=1, is_causal=False,
            )

        x, y = np.asarray(x[:n], dtype=np.float64), np.asarray(y[:n], dtype=np.float64)

        best_p = 1.0
        best_f = 0.0
        best_lag = 1

        for lag in range(1, max_lag + 1):
            f_stat, p_val = self._granger_f_test(x, y, lag)
            if p_val < best_p:
                best_p = p_val
                best_f = f_stat
                best_lag = lag

        return GrangerResult(
            cause="x", effect="y",
            f_statistic=round(best_f, 4),
            p_value=round(best_p, 6),
            optimal_lag=best_lag,
            is_causal=best_p < self.significance,
        )

    def _granger_f_test(self, x: np.ndarray, y: np.ndarray, lag: int) -> tuple[float, float]:
        """Run a single-lag Granger F-test.

        Restricted model:  y_t = a_0 + sum(a_i * y_{t-i})
        Unrestricted model: y_t = a_0 + sum(a_i * y_{t-i}) + sum(b_i * x_{t-i})
        """
        n = len(y)
        # Build lagged matrices
        y_target = y[lag:]
        t = len(y_target)

        # Restricted: only y lags
        restricted_x = np.column_stack(
            [np.ones(t)] + [y[lag - i - 1: n - i - 1] for i in range(lag)]
        )
        # Unrestricted: y lags + x lags
        unrestricted_x = np.column_stack(
            [restricted_x] + [x[lag - i - 1: n - i - 1] for i in range(lag)]
        )

        rss_r = self._ols_rss(restricted_x, y_target)
        rss_u = self._ols_rss(unrestricted_x, y_target)

        df_num = lag  # additional parameters
        df_den = t - unrestricted_x.shape[1]

        if df_den <= 0 or rss_u <= 0:
            return 0.0, 1.0

        f_stat = ((rss_r - rss_u) / df_num) / (rss_u / df_den)
        p_value = 1.0 - stats.f.cdf(f_stat, df_num, df_den)
        return float(f_stat), float(p_value)

    @staticmethod
    def _ols_rss(X: np.ndarray, y: np.ndarray) -> float:
        """Ordinary least squares residual sum of squares."""
        try:
            beta, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
            if len(residuals) > 0:
                return float(residuals[0])
            fitted = X @ beta
            return float(np.sum((y - fitted) ** 2))
        except np.linalg.LinAlgError:
            return float(np.sum(y ** 2))

    # ------------------------------------------------------------------
    # Structural causal model construction
    # ------------------------------------------------------------------

    def build_causal_graph(
        self,
        series_dict: dict[str, np.ndarray],
        max_lag: int | None = None,
    ) -> CausalGraph:
        """Construct a causal graph from multiple asset return series.

        Runs pairwise Granger tests in both directions and retains only
        statistically significant edges.  When both directions are significant,
        the edge with the lower p-value wins (avoids cycles).

        Args:
            series_dict: Mapping of asset name to return series.
            max_lag: Maximum lag for Granger tests.

        Returns:
            CausalGraph with significant directed edges.
        """
        names = list(series_dict.keys())
        edges: list[CausalEdge] = []
        tested_pairs: dict[tuple[str, str], GrangerResult] = {}

        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i == j:
                    continue
                result = self.granger_test(series_dict[a], series_dict[b], max_lag)
                result = GrangerResult(
                    cause=a, effect=b,
                    f_statistic=result.f_statistic,
                    p_value=result.p_value,
                    optimal_lag=result.optimal_lag,
                    is_causal=result.is_causal,
                )
                tested_pairs[(a, b)] = result

        # Resolve bidirectional edges: keep the stronger direction
        added: set[frozenset[str]] = set()
        for (a, b), result in sorted(tested_pairs.items(), key=lambda kv: kv[1].p_value):
            pair_key = frozenset({a, b})
            if pair_key in added:
                continue
            if result.is_causal:
                # Estimate standardized effect size via correlation at optimal lag
                strength = self._lagged_correlation(
                    series_dict[a], series_dict[b], result.optimal_lag
                )
                edges.append(CausalEdge(
                    parent=a, child=b,
                    strength=round(strength, 4),
                    p_value=result.p_value,
                    lag=result.optimal_lag,
                ))
                added.add(pair_key)
                logger.info(
                    "Causal edge: %s -> %s (lag=%d, p=%.4f, strength=%.4f)",
                    a, b, result.optimal_lag, result.p_value, strength,
                )

        adjacency: dict[str, list[str]] = {n: [] for n in names}
        for edge in edges:
            adjacency[edge.parent].append(edge.child)

        graph = CausalGraph(nodes=names, edges=edges, adjacency=adjacency)
        logger.info("Built causal graph: %d nodes, %d edges", len(names), len(edges))
        return graph

    @staticmethod
    def _lagged_correlation(x: np.ndarray, y: np.ndarray, lag: int) -> float:
        """Pearson correlation between x[:-lag] and y[lag:]."""
        if lag <= 0 or lag >= min(len(x), len(y)):
            return 0.0
        r, _ = stats.pearsonr(x[:-lag], y[lag:])
        return float(r)

    # ------------------------------------------------------------------
    # Intervention analysis (do-calculus)
    # ------------------------------------------------------------------

    def intervene(
        self,
        graph: CausalGraph,
        series_dict: dict[str, np.ndarray],
        treatment: str,
        outcome: str,
        treatment_change: float,
    ) -> InterventionResult:
        """Estimate the causal effect of an intervention using do-calculus.

        Attempts the backdoor adjustment first.  Falls back to the frontdoor
        criterion when no valid backdoor set exists, and finally to a direct
        regression estimate.

        Args:
            graph: Pre-built causal graph.
            series_dict: Asset return series.
            treatment: Name of the intervened variable (e.g. "BTC").
            outcome: Name of the outcome variable (e.g. "ETH").
            treatment_change: Size of the intervention (e.g. -0.05 for a 5% drop).

        Returns:
            InterventionResult with estimated causal effect and confidence interval.
        """
        if treatment not in series_dict or outcome not in series_dict:
            logger.error("Treatment or outcome not found in series_dict")
            return InterventionResult(
                treatment=treatment, outcome=outcome,
                treatment_change=treatment_change,
                estimated_effect=0.0,
                confidence_interval=(0.0, 0.0),
                adjustment_set=[], method="none",
            )

        # Try backdoor adjustment
        backdoor_set = self._find_backdoor_set(graph, treatment, outcome)
        if backdoor_set is not None:
            effect, ci = self._backdoor_estimate(
                series_dict, treatment, outcome, backdoor_set, treatment_change,
            )
            return InterventionResult(
                treatment=treatment, outcome=outcome,
                treatment_change=treatment_change,
                estimated_effect=round(effect, 6),
                confidence_interval=(round(ci[0], 6), round(ci[1], 6)),
                adjustment_set=backdoor_set,
                method="backdoor",
            )

        # Try frontdoor adjustment
        frontdoor_set = self._find_frontdoor_set(graph, treatment, outcome)
        if frontdoor_set is not None:
            effect, ci = self._frontdoor_estimate(
                series_dict, treatment, outcome, frontdoor_set, treatment_change,
            )
            return InterventionResult(
                treatment=treatment, outcome=outcome,
                treatment_change=treatment_change,
                estimated_effect=round(effect, 6),
                confidence_interval=(round(ci[0], 6), round(ci[1], 6)),
                adjustment_set=frontdoor_set,
                method="frontdoor",
            )

        # Direct regression fallback
        effect, ci = self._direct_estimate(
            series_dict, treatment, outcome, treatment_change,
        )
        return InterventionResult(
            treatment=treatment, outcome=outcome,
            treatment_change=treatment_change,
            estimated_effect=round(effect, 6),
            confidence_interval=(round(ci[0], 6), round(ci[1], 6)),
            adjustment_set=[],
            method="direct",
        )

    # ------------------------------------------------------------------
    # Backdoor criterion
    # ------------------------------------------------------------------

    def _find_backdoor_set(
        self, graph: CausalGraph, treatment: str, outcome: str,
    ) -> list[str] | None:
        """Find a valid backdoor adjustment set.

        A set Z satisfies the backdoor criterion relative to (treatment, outcome)
        if:
        1. No node in Z is a descendant of treatment.
        2. Z blocks every backdoor path from treatment to outcome
           (paths with an arrow into treatment).

        We use the parents of treatment minus descendants as a candidate set,
        which is a valid (and often minimal) adjustment set when it blocks all
        non-causal paths.
        """
        descendants_of_treatment = graph.descendants(treatment)
        parents_of_treatment = set(graph.parents(treatment))

        # Candidate: parents of treatment that are not descendants
        candidate = [
            p for p in parents_of_treatment
            if p not in descendants_of_treatment and p != outcome
        ]

        if not candidate:
            return None

        # Verify candidate variables exist in data
        return candidate

    def _backdoor_estimate(
        self,
        series_dict: dict[str, np.ndarray],
        treatment: str,
        outcome: str,
        adjustment_set: list[str],
        treatment_change: float,
    ) -> tuple[float, tuple[float, float]]:
        """Estimate causal effect via backdoor adjustment (OLS conditioned on Z).

        Regresses outcome on treatment + adjustment variables, then reads the
        treatment coefficient as the causal effect per unit change.
        """
        n = min(len(series_dict[treatment]), len(series_dict[outcome]))
        for z in adjustment_set:
            n = min(n, len(series_dict[z]))

        y = series_dict[outcome][:n]
        X_cols = [np.ones(n), series_dict[treatment][:n]]
        for z in adjustment_set:
            X_cols.append(series_dict[z][:n])
        X = np.column_stack(X_cols)

        beta, se = self._ols_with_se(X, y)
        # Treatment coefficient is at index 1
        causal_coeff = beta[1]
        coeff_se = se[1]

        effect = causal_coeff * treatment_change
        ci = (
            (causal_coeff - 1.96 * coeff_se) * treatment_change,
            (causal_coeff + 1.96 * coeff_se) * treatment_change,
        )
        logger.info(
            "Backdoor estimate: %s -> %s, coeff=%.4f, effect=%.6f, adj=%s",
            treatment, outcome, causal_coeff, effect, adjustment_set,
        )
        return effect, ci

    # ------------------------------------------------------------------
    # Frontdoor criterion
    # ------------------------------------------------------------------

    def _find_frontdoor_set(
        self, graph: CausalGraph, treatment: str, outcome: str,
    ) -> list[str] | None:
        """Find a valid frontdoor adjustment set.

        A set M satisfies the frontdoor criterion relative to (treatment, outcome)
        if:
        1. Treatment intercepts all directed paths from treatment to outcome
           through M (M are mediators).
        2. There is no unblocked backdoor path from treatment to M.
        3. All backdoor paths from M to outcome are blocked by treatment.

        In practice, we look for direct children of treatment that are also
        direct parents of outcome.
        """
        treatment_children = set(graph.children(treatment))
        outcome_parents = set(graph.parents(outcome))
        mediators = list(treatment_children & outcome_parents)
        return mediators if mediators else None

    def _frontdoor_estimate(
        self,
        series_dict: dict[str, np.ndarray],
        treatment: str,
        outcome: str,
        mediator_set: list[str],
        treatment_change: float,
    ) -> tuple[float, tuple[float, float]]:
        """Estimate causal effect via frontdoor adjustment.

        Two-stage regression:
        Stage 1: mediator ~ treatment  -> coefficient alpha
        Stage 2: outcome ~ mediator (adjusted for treatment) -> coefficient gamma
        Causal effect = alpha * gamma
        """
        n = min(len(series_dict[treatment]), len(series_dict[outcome]))
        mediator_name = mediator_set[0]  # use primary mediator
        n = min(n, len(series_dict[mediator_name]))

        t = series_dict[treatment][:n]
        m = series_dict[mediator_name][:n]
        y = series_dict[outcome][:n]

        # Stage 1: M ~ T
        X1 = np.column_stack([np.ones(n), t])
        beta1, se1 = self._ols_with_se(X1, m)
        alpha = beta1[1]
        alpha_se = se1[1]

        # Stage 2: Y ~ M (controlling for T to satisfy frontdoor condition 3)
        X2 = np.column_stack([np.ones(n), m, t])
        beta2, se2 = self._ols_with_se(X2, y)
        gamma = beta2[1]
        gamma_se = se2[1]

        effect = alpha * gamma * treatment_change
        # Delta method for variance of product
        combined_se = np.sqrt(
            (gamma * alpha_se) ** 2 + (alpha * gamma_se) ** 2
        )
        ci = (
            (alpha * gamma - 1.96 * combined_se) * treatment_change,
            (alpha * gamma + 1.96 * combined_se) * treatment_change,
        )
        logger.info(
            "Frontdoor estimate: %s -> %s via %s, effect=%.6f",
            treatment, outcome, mediator_name, effect,
        )
        return effect, ci

    # ------------------------------------------------------------------
    # Direct estimation fallback
    # ------------------------------------------------------------------

    def _direct_estimate(
        self,
        series_dict: dict[str, np.ndarray],
        treatment: str,
        outcome: str,
        treatment_change: float,
    ) -> tuple[float, tuple[float, float]]:
        """Direct OLS regression without causal adjustment (potentially biased)."""
        n = min(len(series_dict[treatment]), len(series_dict[outcome]))
        X = np.column_stack([np.ones(n), series_dict[treatment][:n]])
        y = series_dict[outcome][:n]

        beta, se = self._ols_with_se(X, y)
        coeff = beta[1]
        coeff_se = se[1]

        effect = coeff * treatment_change
        ci = (
            (coeff - 1.96 * coeff_se) * treatment_change,
            (coeff + 1.96 * coeff_se) * treatment_change,
        )
        logger.warning(
            "Using direct estimate (no valid adjustment set): %s -> %s, coeff=%.4f",
            treatment, outcome, coeff,
        )
        return effect, ci

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _ols_with_se(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """OLS regression returning coefficients and their standard errors."""
        n, k = X.shape
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return np.zeros(k), np.ones(k) * 1e6

        residuals = y - X @ beta
        rss = float(np.sum(residuals ** 2))
        dof = max(n - k, 1)
        sigma2 = rss / dof

        try:
            cov = sigma2 * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        except np.linalg.LinAlgError:
            se = np.ones(k) * 1e6

        return beta, se


causal_engine = CausalInferenceEngine()
