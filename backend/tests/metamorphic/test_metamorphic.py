"""Metamorphic testing for TradeMaster financial calculations.

Metamorphic testing verifies software by checking relationships between
outputs of multiple executions rather than checking individual outputs
against expected values. This is especially valuable for financial
computations where "correct" outputs are hard to determine but
mathematical relationships between transformations are well-defined.

Each test encodes a metamorphic relation (MR) that must hold regardless
of the specific input values.
"""

import math

import numpy as np
import pytest

from app.services.portfolio.pnl import PnLCalculator
from app.services.risk.var import VaRCalculator
from app.services.risk.position_sizer import PositionSizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pnl_series(
    seed: int = 42, n: int = 100, scale: float = 100.0
) -> list[float]:
    """Generate a reproducible P&L series for testing."""
    rng = np.random.RandomState(seed)
    return list(rng.randn(n) * scale)


def _make_returns(
    seed: int = 42, n: int = 200, mu: float = 0.0, sigma: float = 0.02
) -> np.ndarray:
    """Generate a reproducible return series."""
    rng = np.random.RandomState(seed)
    return rng.normal(mu, sigma, n)


def _compute_sharpe(pnl_series: list[float], initial_equity: float) -> float:
    """Convenience wrapper to extract Sharpe ratio."""
    calc = PnLCalculator()
    return calc.calculate_metrics(pnl_series, initial_equity).sharpe_ratio


def _compute_sortino(pnl_series: list[float], initial_equity: float) -> float:
    """Convenience wrapper to extract Sortino ratio."""
    calc = PnLCalculator()
    return calc.calculate_metrics(pnl_series, initial_equity).sortino_ratio


def _compute_max_drawdown(pnl_series: list[float], initial_equity: float) -> float:
    """Convenience wrapper to extract max drawdown."""
    calc = PnLCalculator()
    return calc.calculate_metrics(pnl_series, initial_equity).max_drawdown


def _compute_max_drawdown_pct(pnl_series: list[float], initial_equity: float) -> float:
    """Convenience wrapper to extract max drawdown percentage."""
    calc = PnLCalculator()
    return calc.calculate_metrics(pnl_series, initial_equity).max_drawdown_pct


# ---------------------------------------------------------------------------
# MR1: If investment doubles, absolute total P&L should double
# ---------------------------------------------------------------------------


class TestMR1InvestmentScalingPnL:
    """MR1: Doubling every trade P&L doubles total return."""

    def test_double_pnl_doubles_total_return(self):
        pnls = _make_pnl_series(seed=1)
        calc = PnLCalculator()

        base = calc.calculate_metrics(pnls, 10_000)
        scaled = calc.calculate_metrics([p * 2 for p in pnls], 10_000)

        assert math.isclose(scaled.total_return, base.total_return * 2, rel_tol=1e-9)

    def test_triple_pnl_triples_total_return(self):
        pnls = _make_pnl_series(seed=7)
        calc = PnLCalculator()

        base = calc.calculate_metrics(pnls, 10_000)
        scaled = calc.calculate_metrics([p * 3 for p in pnls], 10_000)

        assert math.isclose(scaled.total_return, base.total_return * 3, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# MR2: Inverting buy/sell (negating P&L) inverts sign of total return
# ---------------------------------------------------------------------------


class TestMR2InvertedPnLSign:
    """MR2: If all trade P&Ls are negated, total return sign flips."""

    def test_negated_pnl_negates_total_return(self):
        pnls = _make_pnl_series(seed=2)
        calc = PnLCalculator()

        base = calc.calculate_metrics(pnls, 10_000)
        inverted = calc.calculate_metrics([-p for p in pnls], 10_000)

        assert math.isclose(inverted.total_return, -base.total_return, rel_tol=1e-9)

    def test_negated_pnl_swaps_win_loss_counts(self):
        pnls = _make_pnl_series(seed=3)
        calc = PnLCalculator()

        base = calc.calculate_metrics(pnls, 10_000)
        inverted = calc.calculate_metrics([-p for p in pnls], 10_000)

        assert inverted.winning_trades == base.losing_trades
        assert inverted.losing_trades == base.winning_trades


# ---------------------------------------------------------------------------
# MR3: Shifting all P&Ls by a constant changes total return linearly
# ---------------------------------------------------------------------------


class TestMR3ConstantShiftPnL:
    """MR3: Adding constant C to each P&L shifts total return by N*C."""

    def test_constant_shift_adds_to_total(self):
        pnls = _make_pnl_series(seed=4, n=50)
        c = 10.0
        calc = PnLCalculator()

        base = calc.calculate_metrics(pnls, 10_000)
        shifted = calc.calculate_metrics([p + c for p in pnls], 10_000)

        expected_shift = len(pnls) * c
        assert math.isclose(
            shifted.total_return, base.total_return + expected_shift, rel_tol=1e-9
        )


# ---------------------------------------------------------------------------
# MR4: Sharpe ratio invariant to portfolio scale (equity scaling)
# ---------------------------------------------------------------------------


class TestMR4SharpeScaleInvariance:
    """MR4: Sharpe ratio remains the same when P&L and equity are scaled
    by the same factor, because returns = pnl / equity are unchanged."""

    def test_sharpe_invariant_when_both_pnl_and_equity_scale(self):
        pnls = _make_pnl_series(seed=5, n=60)
        factor = 5.0
        equity = 10_000.0

        sharpe_base = _compute_sharpe(pnls, equity)
        sharpe_scaled = _compute_sharpe(
            [p * factor for p in pnls], equity * factor
        )

        assert math.isclose(sharpe_base, sharpe_scaled, rel_tol=1e-6)

    def test_sharpe_invariant_half_scale(self):
        pnls = _make_pnl_series(seed=6, n=80)
        factor = 0.5
        equity = 20_000.0

        sharpe_base = _compute_sharpe(pnls, equity)
        sharpe_scaled = _compute_sharpe(
            [p * factor for p in pnls], equity * factor
        )

        assert math.isclose(sharpe_base, sharpe_scaled, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# MR5: VaR scales linearly with position size (portfolio value)
# ---------------------------------------------------------------------------


class TestMR5VaRLinearScaling:
    """MR5: Historical VaR at a given confidence level scales linearly
    with portfolio value."""

    def test_var_doubles_with_portfolio_value(self):
        returns = _make_returns(seed=10, n=200)
        var_calc = VaRCalculator()

        var_base = var_calc.historical_var(returns, 0.95, portfolio_value=10_000)
        var_double = var_calc.historical_var(returns, 0.95, portfolio_value=20_000)

        assert math.isclose(var_double, var_base * 2, rel_tol=1e-9)

    def test_cvar_scales_with_portfolio_value(self):
        returns = _make_returns(seed=11, n=200)
        var_calc = VaRCalculator()

        cvar_base = var_calc.conditional_var(returns, 0.95, portfolio_value=10_000)
        cvar_triple = var_calc.conditional_var(returns, 0.95, portfolio_value=30_000)

        assert math.isclose(cvar_triple, cvar_base * 3, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# MR6: Max drawdown percentage unchanged when returns scaled by positive k
# ---------------------------------------------------------------------------


class TestMR6DrawdownScaleInvariance:
    """MR6: Max drawdown *percentage* is invariant when all P&Ls and the
    initial equity are scaled by the same positive constant, since the
    equity curve shape is preserved proportionally."""

    def test_drawdown_pct_invariant_under_proportional_scaling(self):
        pnls = _make_pnl_series(seed=12, n=80)
        equity = 10_000.0
        factor = 3.0

        dd_pct_base = _compute_max_drawdown_pct(pnls, equity)
        dd_pct_scaled = _compute_max_drawdown_pct(
            [p * factor for p in pnls], equity * factor
        )

        assert math.isclose(dd_pct_base, dd_pct_scaled, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# MR7: Win rate invariant to P&L magnitude (only sign matters)
# ---------------------------------------------------------------------------


class TestMR7WinRateSignOnly:
    """MR7: Win rate depends only on which trades are positive, not
    on their magnitude."""

    def test_win_rate_unchanged_after_scaling(self):
        pnls = _make_pnl_series(seed=13, n=100)
        calc = PnLCalculator()

        base = calc.calculate_metrics(pnls, 10_000)
        scaled = calc.calculate_metrics([p * 7 for p in pnls], 10_000)

        assert base.win_rate == scaled.win_rate
        assert base.winning_trades == scaled.winning_trades
        assert base.losing_trades == scaled.losing_trades


# ---------------------------------------------------------------------------
# MR8: Concatenating identical series does not change Sharpe ratio
# ---------------------------------------------------------------------------


class TestMR8SharpeReplicationInvariance:
    """MR8: Replicating the same P&L series (same returns distribution)
    should yield the same Sharpe ratio within tolerance."""

    def test_sharpe_stable_on_duplicated_series(self):
        pnls = _make_pnl_series(seed=14, n=50)
        equity = 10_000.0

        sharpe_single = _compute_sharpe(pnls, equity)
        sharpe_double = _compute_sharpe(pnls + pnls, equity)

        # Should be very close (not exact because std uses n-1)
        assert math.isclose(sharpe_single, sharpe_double, rel_tol=0.05)


# ---------------------------------------------------------------------------
# MR9: Sortino ratio invariant to portfolio scale
# ---------------------------------------------------------------------------


class TestMR9SortinoScaleInvariance:
    """MR9: Like Sharpe, Sortino ratio is unchanged when P&L and equity
    are scaled by the same factor."""

    def test_sortino_invariant_under_proportional_scaling(self):
        pnls = _make_pnl_series(seed=15, n=80)
        equity = 10_000.0
        factor = 4.0

        sortino_base = _compute_sortino(pnls, equity)
        sortino_scaled = _compute_sortino(
            [p * factor for p in pnls], equity * factor
        )

        if sortino_base == 0 and sortino_scaled == 0:
            return  # Both zero is fine (no downside dev)
        assert math.isclose(sortino_base, sortino_scaled, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# MR10: Position sizing scales linearly with equity
# ---------------------------------------------------------------------------


class TestMR10PositionSizingEquityScaling:
    """MR10: Fixed fraction position size should scale linearly with equity."""

    def test_double_equity_doubles_quantity(self):
        sizer = PositionSizer()
        equity = 10_000.0
        price = 85_000.0
        stop_pct = 0.02

        pos_base = sizer.fixed_fraction(equity, price, stop_pct)
        pos_double = sizer.fixed_fraction(equity * 2, price, stop_pct)

        assert math.isclose(
            pos_double.quantity, pos_base.quantity * 2, rel_tol=1e-9
        )

    def test_risk_amount_scales_with_equity(self):
        sizer = PositionSizer()
        equity = 10_000.0
        price = 85_000.0
        stop_pct = 0.03

        pos_base = sizer.fixed_fraction(equity, price, stop_pct)
        pos_triple = sizer.fixed_fraction(equity * 3, price, stop_pct)

        assert math.isclose(
            pos_triple.risk_amount, pos_base.risk_amount * 3, rel_tol=1e-9
        )


# ---------------------------------------------------------------------------
# MR11: VaR monotonic with confidence level
# ---------------------------------------------------------------------------


class TestMR11VaRMonotonicConfidence:
    """MR11: Higher confidence level should yield higher or equal VaR
    (larger potential loss estimate)."""

    def test_higher_confidence_higher_var(self):
        returns = _make_returns(seed=20, n=300)
        var_calc = VaRCalculator()

        var_90 = var_calc.historical_var(returns, 0.90, 10_000)
        var_95 = var_calc.historical_var(returns, 0.95, 10_000)
        var_99 = var_calc.historical_var(returns, 0.99, 10_000)

        assert var_95 >= var_90 - 1e-9
        assert var_99 >= var_95 - 1e-9


# ---------------------------------------------------------------------------
# MR12: CVaR >= VaR at the same confidence level
# ---------------------------------------------------------------------------


class TestMR12CVaRGreaterThanVaR:
    """MR12: Conditional VaR (Expected Shortfall) is always >= VaR."""

    def test_cvar_at_least_as_large_as_var(self):
        returns = _make_returns(seed=21, n=300)
        var_calc = VaRCalculator()

        for conf in [0.90, 0.95, 0.99]:
            var = var_calc.historical_var(returns, conf, 10_000)
            cvar = var_calc.conditional_var(returns, conf, 10_000)
            assert cvar >= var - 1e-9, (
                f"CVaR ({cvar}) < VaR ({var}) at confidence {conf}"
            )


# ---------------------------------------------------------------------------
# MR13: Profit factor invariant to uniform P&L scaling
# ---------------------------------------------------------------------------


class TestMR13ProfitFactorScaleInvariance:
    """MR13: Profit factor (gross_profit / gross_loss) is invariant
    when all P&Ls are multiplied by a positive constant."""

    def test_profit_factor_unchanged_after_scaling(self):
        pnls = _make_pnl_series(seed=22, n=60)
        calc = PnLCalculator()

        base = calc.calculate_metrics(pnls, 10_000)
        scaled = calc.calculate_metrics([p * 5 for p in pnls], 10_000)

        if math.isinf(base.profit_factor) and math.isinf(scaled.profit_factor):
            return  # Both inf is fine (no losses)
        assert math.isclose(base.profit_factor, scaled.profit_factor, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# MR14: Expectancy scales linearly with P&L magnitude
# ---------------------------------------------------------------------------


class TestMR14ExpectancyLinearScaling:
    """MR14: Expectancy (average P&L per trade) scales linearly with
    a constant multiplier applied to all P&Ls."""

    def test_expectancy_doubles_when_pnl_doubles(self):
        pnls = _make_pnl_series(seed=23, n=50)
        calc = PnLCalculator()

        base = calc.calculate_metrics(pnls, 10_000)
        scaled = calc.calculate_metrics([p * 2 for p in pnls], 10_000)

        assert math.isclose(scaled.expectancy, base.expectancy * 2, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# MR15: Kelly criterion: zero edge yields zero position
# ---------------------------------------------------------------------------


class TestMR15KellyZeroEdge:
    """MR15: When win_rate = 0.5 and avg_win == avg_loss (no edge),
    Kelly fraction is zero, so position size should be zero."""

    def test_no_edge_gives_zero_position(self):
        sizer = PositionSizer()
        pos = sizer.fractional_kelly(
            equity=10_000,
            win_rate=0.5,
            avg_win=100,
            avg_loss=100,
            price=85_000,
            stop_distance_pct=0.02,
        )
        assert pos.quantity == 0.0
        assert pos.notional_value == 0.0

    def test_negative_expectancy_gives_zero_position(self):
        sizer = PositionSizer()
        pos = sizer.fractional_kelly(
            equity=10_000,
            win_rate=0.3,
            avg_win=100,
            avg_loss=100,
            price=85_000,
            stop_distance_pct=0.02,
        )
        assert pos.quantity == 0.0


# ---------------------------------------------------------------------------
# MR16: Reversing the order of P&Ls does not change aggregate metrics
# ---------------------------------------------------------------------------


class TestMR16OrderIndependence:
    """MR16: Total return, win rate, and expectancy are independent
    of the order of trades. Sharpe (depends on sequence via equity curve)
    may differ, but sum-based metrics must not."""

    def test_reversed_pnl_same_totals(self):
        pnls = _make_pnl_series(seed=24, n=60)
        calc = PnLCalculator()

        base = calc.calculate_metrics(pnls, 10_000)
        rev = calc.calculate_metrics(list(reversed(pnls)), 10_000)

        assert math.isclose(base.total_return, rev.total_return, rel_tol=1e-9)
        assert base.winning_trades == rev.winning_trades
        assert base.losing_trades == rev.losing_trades
        assert math.isclose(base.win_rate, rev.win_rate, rel_tol=1e-9)
        assert math.isclose(base.expectancy, rev.expectancy, rel_tol=1e-9)
        assert math.isclose(base.profit_factor, rev.profit_factor, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# MR17: Parametric VaR scales with sqrt of holding period
# ---------------------------------------------------------------------------


class TestMR17ParametricVaRSqrtTime:
    """MR17: Parametric VaR scales with sqrt(holding_period_days)."""

    def test_var_scales_sqrt_time(self):
        var_calc = VaRCalculator()
        vol = 0.02
        pv = 100_000

        var_1d = var_calc.parametric_var(pv, vol, 0.95, holding_period_days=1)
        var_4d = var_calc.parametric_var(pv, vol, 0.95, holding_period_days=4)

        # var_4d should be 2x var_1d (sqrt(4) = 2)
        assert math.isclose(var_4d, var_1d * 2, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# MR18: Position sizer respects max exposure cap
# ---------------------------------------------------------------------------


class TestMR18PositionSizerExposureCap:
    """MR18: Regardless of inputs, notional value never exceeds
    max_single_asset_exposure * equity."""

    @pytest.mark.parametrize("equity", [1_000, 10_000, 100_000, 1_000_000])
    def test_notional_capped(self, equity):
        sizer = PositionSizer(
            max_risk_per_trade=0.02,
            max_single_asset_exposure=0.30,
        )
        # Very tight stop => large notional before cap
        pos = sizer.fixed_fraction(equity, price=100.0, stop_distance_pct=0.001)
        max_allowed = equity * sizer.max_single_asset_exposure
        assert pos.notional_value <= max_allowed + 1e-9
