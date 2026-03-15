"""Unit tests for PnL metrics with hand-calculated reference values.

Each test validates against manually computed values to ensure
Sharpe, Sortino, Calmar, and drawdown formulas are correct.
"""

import math

import numpy as np
import pytest

from app.services.portfolio.pnl import PnLCalculator


@pytest.fixture
def calc():
    return PnLCalculator()


class TestSharpeRatio:
    """Sharpe = (avg_return - rf/252) / std(returns) * sqrt(252)"""

    def test_positive_sharpe_no_rf(self, calc):
        # 5 trades: +100, +50, -30, +80, +40 on $10000
        pnls = [100.0, 50.0, -30.0, 80.0, 40.0]
        m = calc.calculate_metrics(pnls, initial_equity=10000.0, risk_free_rate=0.0)

        returns = np.array([0.01, 0.005, -0.003, 0.008, 0.004])
        avg_r = returns.mean()  # 0.0048
        std_r = returns.std()   # ~0.00445
        expected = (avg_r / std_r) * math.sqrt(252)  # ~17.13

        assert abs(m.sharpe_ratio - expected) < 0.01

    def test_sharpe_with_rf(self, calc):
        pnls = [100.0, 50.0, -30.0, 80.0, 40.0]
        rf = 0.05  # 5% annual
        m = calc.calculate_metrics(pnls, initial_equity=10000.0, risk_free_rate=rf)

        returns = np.array([0.01, 0.005, -0.003, 0.008, 0.004])
        daily_rf = 0.05 / 252
        avg_excess = returns.mean() - daily_rf
        std_r = returns.std()
        expected = (avg_excess / std_r) * math.sqrt(252)

        assert abs(m.sharpe_ratio - expected) < 0.01

    def test_all_same_returns_sharpe_zero(self, calc):
        # Zero std -> sharpe = 0
        pnls = [50.0, 50.0, 50.0, 50.0]
        m = calc.calculate_metrics(pnls, initial_equity=10000.0)
        assert m.sharpe_ratio == 0

    def test_negative_returns_negative_sharpe(self, calc):
        pnls = [-100.0, -50.0, -80.0, -60.0]
        m = calc.calculate_metrics(pnls, initial_equity=10000.0)
        assert m.sharpe_ratio < 0


class TestSortinoRatio:
    """Sortino = (avg_return - rf/252) / sqrt(mean(min(r-rf, 0)^2)) * sqrt(252)"""

    def test_sortino_basic(self, calc):
        pnls = [100.0, -50.0, 80.0, -20.0, 60.0]
        m = calc.calculate_metrics(pnls, initial_equity=10000.0, risk_free_rate=0.0)

        returns = np.array(pnls) / 10000.0
        downside = np.minimum(returns, 0)
        dd = math.sqrt(np.mean(downside ** 2))
        expected = (returns.mean() / dd) * math.sqrt(252) if dd > 0 else 0

        assert abs(m.sortino_ratio - expected) < 0.01

    def test_sortino_all_positive_returns(self, calc):
        # All positive returns -> downside dev = 0 -> sortino = 0
        pnls = [100.0, 50.0, 80.0, 60.0]
        m = calc.calculate_metrics(pnls, initial_equity=10000.0)
        assert m.sortino_ratio == 0

    def test_sortino_with_rf(self, calc):
        pnls = [100.0, -50.0, 80.0, -20.0, 60.0]
        rf = 0.05
        m = calc.calculate_metrics(pnls, initial_equity=10000.0, risk_free_rate=rf)

        returns = np.array(pnls) / 10000.0
        daily_rf = rf / 252
        downside = np.minimum(returns - daily_rf, 0)
        dd = math.sqrt(np.mean(downside ** 2))
        expected = ((returns.mean() - daily_rf) / dd) * math.sqrt(252) if dd > 0 else 0

        assert abs(m.sortino_ratio - expected) < 0.01


class TestCalmarRatio:
    """Calmar = annualized_return / max_drawdown_pct"""

    def test_calmar_basic(self, calc):
        # 252 trades (1 year), total P&L = +2000 on $10000 = 20%
        # Already annualized for 1 year: annualized_return = 20%
        pnls = [2000.0 / 252] * 252
        m = calc.calculate_metrics(pnls, initial_equity=10000.0)

        # No drawdown with constant positive returns -> calmar = 0 (0 dd)
        assert m.calmar_ratio == 0  # Can't divide by 0 drawdown

    def test_calmar_with_drawdown(self, calc):
        # Simulate: up 500, down 300, up 800 on $10000
        pnls = [500.0, -300.0, 800.0]
        m = calc.calculate_metrics(pnls, initial_equity=10000.0)

        total_return = 1000.0 / 10000.0  # 10%
        years = 3 / 252
        annualized = (1 + total_return) ** (1 / years) - 1

        # Drawdown: equity goes 10500 -> 10200 -> 11000
        # Peak at 10500, dd = 300, dd_pct = 300/10500
        dd_pct = 300.0 / 10500.0
        expected_calmar = annualized / dd_pct

        assert abs(m.calmar_ratio - expected_calmar) < 0.1

    def test_calmar_no_loss(self, calc):
        pnls = [100.0, 200.0, 300.0]
        m = calc.calculate_metrics(pnls, initial_equity=10000.0)
        assert m.calmar_ratio == 0  # No drawdown


class TestMaxDrawdown:

    def test_drawdown_simple(self, calc):
        # Equity: 10000 -> 10500 -> 10200 -> 11000
        pnls = [500.0, -300.0, 800.0]
        m = calc.calculate_metrics(pnls, initial_equity=10000.0)

        assert abs(m.max_drawdown - 300.0) < 0.01
        assert abs(m.max_drawdown_pct - 300.0 / 10500.0) < 0.001

    def test_drawdown_deeper_later(self, calc):
        # Equity: 10000 -> 10500 -> 10200 -> 10800 -> 9800
        pnls = [500.0, -300.0, 600.0, -1000.0]
        m = calc.calculate_metrics(pnls, initial_equity=10000.0)

        # Peak before last loss: 10800, dd = 1000
        assert abs(m.max_drawdown - 1000.0) < 0.01
        assert abs(m.max_drawdown_pct - 1000.0 / 10800.0) < 0.001

    def test_no_drawdown(self, calc):
        pnls = [100.0, 200.0, 300.0]
        m = calc.calculate_metrics(pnls, initial_equity=10000.0)
        assert m.max_drawdown == 0
        assert m.max_drawdown_pct == 0


class TestBasicMetrics:

    def test_win_rate(self, calc):
        pnls = [100.0, -50.0, 80.0, -20.0, 60.0]
        m = calc.calculate_metrics(pnls, initial_equity=10000.0)
        assert m.win_rate == 3 / 5
        assert m.winning_trades == 3
        assert m.losing_trades == 2

    def test_profit_factor(self, calc):
        pnls = [100.0, -50.0, 80.0, -20.0]
        m = calc.calculate_metrics(pnls, initial_equity=10000.0)
        # Gross profit = 180, Gross loss = 70
        assert abs(m.profit_factor - 180.0 / 70.0) < 0.001

    def test_empty_series(self, calc):
        m = calc.calculate_metrics([], initial_equity=10000.0)
        assert m.total_trades == 0
        assert m.sharpe_ratio == 0
        assert m.sortino_ratio == 0
        assert m.calmar_ratio == 0

    def test_expectancy(self, calc):
        pnls = [100.0, -50.0, 80.0]
        m = calc.calculate_metrics(pnls, initial_equity=10000.0)
        assert abs(m.expectancy - (100 - 50 + 80) / 3) < 0.01
