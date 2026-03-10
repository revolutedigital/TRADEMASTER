"""Differential fuzzing oracle tests for TradeMaster financial calculations.

Strategy: implement each financial metric twice -- once as a simple/naive
"reference oracle" and once using the production code.  Hypothesis generates
random inputs and we compare both outputs.  Any divergence signals a bug in
either implementation (or a specification misunderstanding).

Divergences are reported with fully reproducible inputs via Hypothesis's
database and shrinking.
"""

import math
from typing import NamedTuple

import numpy as np
import pytest

try:
    from hypothesis import given, strategies as st, settings as hyp_settings, assume, note
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    pytest.skip("hypothesis not installed", allow_module_level=True)

from app.services.portfolio.pnl import PnLCalculator
from app.services.risk.var import VaRCalculator
from app.services.risk.position_sizer import PositionSizer


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

# Bounded returns that avoid extreme float issues
_return_value = st.floats(min_value=-0.20, max_value=0.20)

_pnl_value = st.floats(min_value=-5_000.0, max_value=5_000.0)

_pnl_list = st.lists(_pnl_value, min_size=5, max_size=200)

_returns_list = st.lists(
    _return_value, min_size=20, max_size=500
)

_positive_float = st.floats(min_value=1.0, max_value=1_000_000.0)


# ---------------------------------------------------------------------------
# Reference oracle implementations (simple / textbook)
# ---------------------------------------------------------------------------


def _ref_sharpe_ratio(
    pnl_series: list[float], initial_equity: float
) -> float:
    """Naive Sharpe ratio: annualised mean / std of per-trade returns."""
    if len(pnl_series) < 2 or initial_equity <= 0:
        return 0.0
    returns = [p / initial_equity for p in pnl_series]
    mean_r = sum(returns) / len(returns)
    var = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
    std_r = math.sqrt(var) if var > 0 else 0.0
    if std_r == 0:
        return 0.0
    return (mean_r / std_r) * math.sqrt(252)


def _ref_sortino_ratio(
    pnl_series: list[float], initial_equity: float
) -> float:
    """Naive Sortino ratio: annualised mean / downside deviation."""
    if len(pnl_series) < 2 or initial_equity <= 0:
        return 0.0
    returns = [p / initial_equity for p in pnl_series]
    mean_r = sum(returns) / len(returns)
    neg_returns = [r for r in returns if r < 0]
    if len(neg_returns) < 2:
        return 0.0
    neg_mean = sum(neg_returns) / len(neg_returns)
    neg_var = sum((r - neg_mean) ** 2 for r in neg_returns) / (len(neg_returns) - 1)
    downside_std = math.sqrt(neg_var) if neg_var > 0 else 0.0
    if downside_std == 0:
        return 0.0
    return (mean_r / downside_std) * math.sqrt(252)


def _ref_max_drawdown(
    pnl_series: list[float], initial_equity: float
) -> float:
    """Naive max drawdown in absolute terms."""
    equity = initial_equity
    peak = equity
    max_dd = 0.0
    for pnl in pnl_series:
        equity += pnl
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _ref_max_drawdown_pct(
    pnl_series: list[float], initial_equity: float
) -> float:
    """Naive max drawdown percentage."""
    equity = initial_equity
    peak = equity
    max_dd_pct = 0.0
    for pnl in pnl_series:
        equity += pnl
        if equity > peak:
            peak = equity
        if peak > 0:
            dd_pct = (peak - equity) / peak
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
    return max_dd_pct


def _ref_historical_var(
    returns: list[float], confidence: float, portfolio_value: float
) -> float:
    """Naive historical VaR using sorted list."""
    if len(returns) < 10:
        return 0.0
    sorted_r = sorted(returns)
    idx = int(math.floor((1 - confidence) * len(sorted_r)))
    idx = max(0, min(idx, len(sorted_r) - 1))
    return abs(sorted_r[idx] * portfolio_value)


def _ref_conditional_var(
    returns: list[float], confidence: float, portfolio_value: float
) -> float:
    """Naive CVaR: average of returns below VaR threshold."""
    if len(returns) < 10:
        return 0.0
    sorted_r = sorted(returns)
    cutoff_idx = int(math.floor((1 - confidence) * len(sorted_r)))
    cutoff_idx = max(1, min(cutoff_idx, len(sorted_r) - 1))
    threshold = sorted_r[cutoff_idx]
    tail = [r for r in sorted_r if r <= threshold]
    if not tail:
        return abs(threshold * portfolio_value)
    avg_tail = sum(tail) / len(tail)
    return abs(avg_tail * portfolio_value)


def _ref_fixed_fraction_quantity(
    equity: float,
    price: float,
    stop_distance_pct: float,
    max_risk: float = 0.02,
    max_exposure: float = 0.30,
) -> float:
    """Naive fixed-fraction position size."""
    if equity <= 0 or price <= 0 or stop_distance_pct <= 0:
        return 0.0
    risk_amount = equity * max_risk
    notional = risk_amount / stop_distance_pct
    max_notional = equity * max_exposure
    notional = min(notional, max_notional)
    return notional / price


# ---------------------------------------------------------------------------
# Differential tests
# ---------------------------------------------------------------------------


class TestDifferentialSharpeRatio:
    """Compare production Sharpe ratio against reference oracle."""

    @given(pnls=_pnl_list, equity=_positive_float)
    @hyp_settings(max_examples=300, deadline=None)
    def test_sharpe_matches_reference(self, pnls, equity):
        assume(len(pnls) >= 5)
        assume(not all(p == 0 for p in pnls))

        prod_calc = PnLCalculator()
        prod_sharpe = prod_calc.calculate_metrics(pnls, equity).sharpe_ratio
        ref_sharpe = _ref_sharpe_ratio(pnls, equity)

        # Both use slightly different numpy vs pure-python std (ddof).
        # numpy.std defaults to ddof=0 while reference uses ddof=1.
        # Allow small relative tolerance due to this difference.
        if ref_sharpe == 0 and prod_sharpe == 0:
            return
        note(f"prod={prod_sharpe}, ref={ref_sharpe}, n={len(pnls)}")
        # The production code uses np.std (ddof=0) while reference uses ddof=1.
        # Ratio between them: sqrt(n/(n-1)). Verify they agree up to that factor.
        n = len(pnls)
        correction = math.sqrt(n / (n - 1))
        assert math.isclose(prod_sharpe, ref_sharpe / correction, rel_tol=0.01) or \
               math.isclose(prod_sharpe, ref_sharpe, rel_tol=0.05), (
            f"Sharpe divergence: prod={prod_sharpe}, ref={ref_sharpe}, "
            f"equity={equity}, pnls[:5]={pnls[:5]}"
        )


class TestDifferentialSortinoRatio:
    """Compare production Sortino ratio against reference oracle."""

    @given(pnls=_pnl_list, equity=_positive_float)
    @hyp_settings(max_examples=300, deadline=None)
    def test_sortino_matches_reference(self, pnls, equity):
        assume(len(pnls) >= 10)
        # Ensure we have some negative P&Ls for meaningful Sortino
        neg_count = sum(1 for p in pnls if p < 0)
        assume(neg_count >= 3)

        prod_calc = PnLCalculator()
        prod_sortino = prod_calc.calculate_metrics(pnls, equity).sortino_ratio
        ref_sortino = _ref_sortino_ratio(pnls, equity)

        if ref_sortino == 0 and prod_sortino == 0:
            return
        note(f"prod={prod_sortino}, ref={ref_sortino}")
        # Same ddof correction as Sharpe
        neg_n = neg_count
        if neg_n > 1:
            correction = math.sqrt(neg_n / (neg_n - 1))
            assert math.isclose(prod_sortino, ref_sortino / correction, rel_tol=0.02) or \
                   math.isclose(prod_sortino, ref_sortino, rel_tol=0.10), (
                f"Sortino divergence: prod={prod_sortino}, ref={ref_sortino}"
            )


class TestDifferentialMaxDrawdown:
    """Compare production max drawdown against reference oracle."""

    @given(pnls=_pnl_list, equity=_positive_float)
    @hyp_settings(max_examples=300, deadline=None)
    def test_max_drawdown_matches_reference(self, pnls, equity):
        assume(len(pnls) >= 5)

        prod_calc = PnLCalculator()
        prod_dd = prod_calc.calculate_metrics(pnls, equity).max_drawdown
        ref_dd = _ref_max_drawdown(pnls, equity)

        assert math.isclose(prod_dd, ref_dd, rel_tol=1e-9, abs_tol=1e-9), (
            f"Max drawdown divergence: prod={prod_dd}, ref={ref_dd}, "
            f"equity={equity}, pnls[:5]={pnls[:5]}"
        )

    @given(pnls=_pnl_list, equity=_positive_float)
    @hyp_settings(max_examples=300, deadline=None)
    def test_max_drawdown_pct_matches_reference(self, pnls, equity):
        assume(len(pnls) >= 5)

        prod_calc = PnLCalculator()
        prod_dd_pct = prod_calc.calculate_metrics(pnls, equity).max_drawdown_pct
        ref_dd_pct = _ref_max_drawdown_pct(pnls, equity)

        assert math.isclose(prod_dd_pct, ref_dd_pct, rel_tol=1e-6, abs_tol=1e-9), (
            f"Max drawdown % divergence: prod={prod_dd_pct}, ref={ref_dd_pct}"
        )


class TestDifferentialHistoricalVaR:
    """Compare production VaR against reference oracle."""

    @given(
        returns=_returns_list,
        confidence=st.sampled_from([0.90, 0.95, 0.99]),
        portfolio_value=_positive_float,
    )
    @hyp_settings(max_examples=300, deadline=None)
    def test_historical_var_matches_reference(self, returns, confidence, portfolio_value):
        assume(len(returns) >= 20)

        var_calc = VaRCalculator()
        prod_var = var_calc.historical_var(returns, confidence, portfolio_value)
        ref_var = _ref_historical_var(returns, confidence, portfolio_value)

        # np.percentile uses interpolation; reference uses floor index.
        # Allow some tolerance for the interpolation difference.
        if ref_var == 0 and prod_var == 0:
            return
        note(f"prod={prod_var}, ref={ref_var}, n={len(returns)}")
        assert math.isclose(prod_var, ref_var, rel_tol=0.15, abs_tol=1.0), (
            f"VaR divergence: prod={prod_var}, ref={ref_var}, "
            f"conf={confidence}, pv={portfolio_value}"
        )


class TestDifferentialConditionalVaR:
    """Compare production CVaR against reference oracle."""

    @given(
        returns=_returns_list,
        confidence=st.sampled_from([0.90, 0.95]),
        portfolio_value=_positive_float,
    )
    @hyp_settings(max_examples=200, deadline=None)
    def test_cvar_matches_reference(self, returns, confidence, portfolio_value):
        assume(len(returns) >= 30)

        var_calc = VaRCalculator()
        prod_cvar = var_calc.conditional_var(returns, confidence, portfolio_value)
        ref_cvar = _ref_conditional_var(returns, confidence, portfolio_value)

        if ref_cvar == 0 and prod_cvar == 0:
            return
        note(f"prod={prod_cvar}, ref={ref_cvar}")
        assert math.isclose(prod_cvar, ref_cvar, rel_tol=0.20, abs_tol=1.0), (
            f"CVaR divergence: prod={prod_cvar}, ref={ref_cvar}"
        )


class TestDifferentialPositionSizing:
    """Compare production position sizing against reference oracle."""

    @given(
        equity=st.floats(min_value=1_000, max_value=10_000_000),
        price=st.floats(min_value=0.01, max_value=500_000),
        stop_pct=st.floats(min_value=0.001, max_value=0.20),
    )
    @hyp_settings(max_examples=300, deadline=None)
    def test_fixed_fraction_matches_reference(self, equity, price, stop_pct):
        sizer = PositionSizer()
        prod_pos = sizer.fixed_fraction(equity, price, stop_pct)
        ref_qty = _ref_fixed_fraction_quantity(
            equity, price, stop_pct,
            max_risk=sizer.max_risk_per_trade,
            max_exposure=sizer.max_single_asset_exposure,
        )

        assert math.isclose(prod_pos.quantity, ref_qty, rel_tol=1e-9, abs_tol=1e-12), (
            f"Position sizing divergence: prod={prod_pos.quantity}, ref={ref_qty}, "
            f"equity={equity}, price={price}, stop_pct={stop_pct}"
        )

    @given(
        equity=st.floats(min_value=1_000, max_value=10_000_000),
        price=st.floats(min_value=0.01, max_value=500_000),
        stop_pct=st.floats(min_value=0.001, max_value=0.20),
    )
    @hyp_settings(max_examples=200, deadline=None)
    def test_notional_never_exceeds_max_exposure(self, equity, price, stop_pct):
        """Invariant: notional_value <= max_single_asset_exposure * equity."""
        sizer = PositionSizer()
        pos = sizer.fixed_fraction(equity, price, stop_pct)
        max_allowed = equity * sizer.max_single_asset_exposure
        assert pos.notional_value <= max_allowed + 1e-6, (
            f"Exposure breach: notional={pos.notional_value}, max={max_allowed}"
        )

    @given(
        equity=st.floats(min_value=1_000, max_value=10_000_000),
        price=st.floats(min_value=0.01, max_value=500_000),
        stop_pct=st.floats(min_value=0.001, max_value=0.20),
    )
    @hyp_settings(max_examples=200, deadline=None)
    def test_risk_amount_never_exceeds_max_risk(self, equity, price, stop_pct):
        """Invariant: risk_amount <= max_risk_per_trade * equity."""
        sizer = PositionSizer()
        pos = sizer.fixed_fraction(equity, price, stop_pct)
        max_risk_amt = equity * sizer.max_risk_per_trade
        assert pos.risk_amount <= max_risk_amt + 1e-6, (
            f"Risk breach: risk_amount={pos.risk_amount}, max={max_risk_amt}"
        )


class TestDifferentialEdgeCases:
    """Verify both implementations handle edge cases identically."""

    def test_empty_pnl_series(self):
        calc = PnLCalculator()
        metrics = calc.calculate_metrics([], 10_000)
        assert metrics.total_return == 0
        assert metrics.sharpe_ratio == 0
        assert metrics.max_drawdown == 0

    def test_all_winning_trades(self):
        pnls = [100.0, 200.0, 50.0, 300.0, 150.0]
        calc = PnLCalculator()
        metrics = calc.calculate_metrics(pnls, 10_000)
        ref_dd = _ref_max_drawdown(pnls, 10_000)

        assert metrics.win_rate == 1.0
        assert metrics.losing_trades == 0
        assert metrics.max_drawdown == ref_dd
        assert ref_dd == 0.0  # All positive => no drawdown

    def test_all_losing_trades(self):
        pnls = [-100.0, -200.0, -50.0, -300.0, -150.0]
        calc = PnLCalculator()
        metrics = calc.calculate_metrics(pnls, 10_000)
        ref_dd = _ref_max_drawdown(pnls, 10_000)

        assert metrics.win_rate == 0.0
        assert metrics.winning_trades == 0
        assert math.isclose(metrics.max_drawdown, ref_dd, rel_tol=1e-9)

    def test_single_trade(self):
        calc = PnLCalculator()
        metrics = calc.calculate_metrics([500.0], 10_000)
        assert metrics.total_trades == 1
        assert metrics.total_return == 500.0
        assert metrics.winning_trades == 1

    def test_var_insufficient_data(self):
        var_calc = VaRCalculator()
        short_returns = [0.01, -0.02, 0.005]
        assert var_calc.historical_var(short_returns, 0.95, 10_000) == 0.0
        assert var_calc.conditional_var(short_returns, 0.95, 10_000) == 0.0
        assert _ref_historical_var(short_returns, 0.95, 10_000) == 0.0

    def test_zero_equity_position_sizing(self):
        sizer = PositionSizer()
        pos = sizer.fixed_fraction(0, 100.0, 0.02)
        ref = _ref_fixed_fraction_quantity(0, 100.0, 0.02)
        assert pos.quantity == 0.0
        assert ref == 0.0

    def test_zero_price_position_sizing(self):
        sizer = PositionSizer()
        pos = sizer.fixed_fraction(10_000, 0, 0.02)
        ref = _ref_fixed_fraction_quantity(10_000, 0, 0.02)
        assert pos.quantity == 0.0
        assert ref == 0.0
