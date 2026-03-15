"""Property-based tests for financial calculation invariants.

Uses hypothesis to generate random inputs and verify that mathematical
properties always hold regardless of input values.
"""
import math
import pytest

try:
    from hypothesis import given, strategies as st, settings as hyp_settings, assume
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    # Create dummy decorators so tests are skipped cleanly
    class _st:
        @staticmethod
        def floats(*a, **kw): return None
        @staticmethod
        def lists(*a, **kw): return None
        @staticmethod
        def integers(*a, **kw): return None
    st = _st()
    def given(*a, **kw):
        def decorator(f):
            return pytest.mark.skip(reason="hypothesis not installed")(f)
        return decorator
    def assume(x): pass
    class hyp_settings:
        def __init__(self, **kw): pass
        def __call__(self, f): return f

import numpy as np
from app.services.portfolio.pnl import PnLCalculator


calc = PnLCalculator()


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestPnLInvariants:

    @given(pnls=st.lists(
        st.floats(min_value=-10000, max_value=10000, allow_nan=False, allow_infinity=False),
        min_size=1, max_size=500,
    ))
    @hyp_settings(max_examples=200)
    def test_total_return_equals_sum(self, pnls):
        """Total return must always equal sum of individual P&Ls."""
        m = calc.calculate_metrics(pnls, initial_equity=10000.0)
        expected = sum(pnls)
        assert abs(m.total_return - expected) < 0.01

    @given(pnls=st.lists(
        st.floats(min_value=-10000, max_value=10000, allow_nan=False, allow_infinity=False),
        min_size=1, max_size=500,
    ))
    @hyp_settings(max_examples=200)
    def test_win_rate_bounded(self, pnls):
        """Win rate must always be between 0 and 1."""
        m = calc.calculate_metrics(pnls, initial_equity=10000.0)
        assert 0 <= m.win_rate <= 1

    @given(pnls=st.lists(
        st.floats(min_value=-10000, max_value=10000, allow_nan=False, allow_infinity=False),
        min_size=1, max_size=500,
    ))
    @hyp_settings(max_examples=200)
    def test_win_plus_loss_equals_total(self, pnls):
        """Winning + losing trades should account for all non-zero trades."""
        m = calc.calculate_metrics(pnls, initial_equity=10000.0)
        # Zero P&L trades are neither win nor loss
        zero_trades = sum(1 for p in pnls if p == 0)
        assert m.winning_trades + m.losing_trades + zero_trades == m.total_trades

    @given(pnls=st.lists(
        st.floats(min_value=-10000, max_value=10000, allow_nan=False, allow_infinity=False),
        min_size=1, max_size=500,
    ))
    @hyp_settings(max_examples=200)
    def test_max_drawdown_non_negative(self, pnls):
        """Max drawdown must always be >= 0."""
        m = calc.calculate_metrics(pnls, initial_equity=10000.0)
        assert m.max_drawdown >= 0
        assert m.max_drawdown_pct >= 0

    @given(pnls=st.lists(
        st.floats(min_value=-10000, max_value=10000, allow_nan=False, allow_infinity=False),
        min_size=2, max_size=500,
    ))
    @hyp_settings(max_examples=200)
    def test_sharpe_never_nan(self, pnls):
        """Sharpe ratio must never be NaN."""
        m = calc.calculate_metrics(pnls, initial_equity=10000.0)
        assert not math.isnan(m.sharpe_ratio)

    @given(pnls=st.lists(
        st.floats(min_value=-10000, max_value=10000, allow_nan=False, allow_infinity=False),
        min_size=2, max_size=500,
    ))
    @hyp_settings(max_examples=200)
    def test_sortino_never_nan(self, pnls):
        """Sortino ratio must never be NaN."""
        m = calc.calculate_metrics(pnls, initial_equity=10000.0)
        assert not math.isnan(m.sortino_ratio)

    @given(
        pnls=st.lists(
            st.floats(min_value=0.01, max_value=10000, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=100,
        ),
        multiplier=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @hyp_settings(max_examples=100)
    def test_scaling_preserves_ratios(self, pnls, multiplier):
        """Scaling all P&Ls and equity by same factor should preserve ratios."""
        m1 = calc.calculate_metrics(pnls, initial_equity=10000.0)
        scaled_pnls = [p * multiplier for p in pnls]
        m2 = calc.calculate_metrics(scaled_pnls, initial_equity=10000.0 * multiplier)

        # Win rate should be identical
        assert abs(m1.win_rate - m2.win_rate) < 0.001
        # Return % should be identical
        assert abs(m1.total_return_pct - m2.total_return_pct) < 0.001


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestRiskInvariants:

    @given(pnls=st.lists(
        st.floats(min_value=0.01, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=2, max_size=100,
    ))
    @hyp_settings(max_examples=100)
    def test_all_positive_no_drawdown(self, pnls):
        """All positive P&Ls should result in zero drawdown."""
        m = calc.calculate_metrics(pnls, initial_equity=10000.0)
        assert m.max_drawdown == 0
        assert m.max_drawdown_pct == 0

    @given(pnls=st.lists(
        st.floats(min_value=-1000, max_value=-0.01, allow_nan=False, allow_infinity=False),
        min_size=2, max_size=100,
    ))
    @hyp_settings(max_examples=100)
    def test_all_negative_has_drawdown(self, pnls):
        """All negative P&Ls should always have a drawdown."""
        m = calc.calculate_metrics(pnls, initial_equity=10000.0)
        assert m.max_drawdown > 0

    @given(pnls=st.lists(
        st.floats(min_value=-1000, max_value=-0.01, allow_nan=False, allow_infinity=False),
        min_size=2, max_size=100,
    ))
    @hyp_settings(max_examples=100)
    def test_negative_sharpe_for_all_losses(self, pnls):
        """All negative P&Ls should produce negative Sharpe (with rf=0)."""
        m = calc.calculate_metrics(pnls, initial_equity=10000.0, risk_free_rate=0.0)
        assert m.sharpe_ratio <= 0
