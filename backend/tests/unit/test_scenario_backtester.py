"""Unit tests for scenario backtesting and systemic risk."""
import numpy as np
import pytest

from app.services.risk.scenario_backtester import ScenarioBacktester
from app.services.risk.systemic import SystemicRiskMonitor
from app.services.risk.hedging import HedgingSuggester


class TestScenarioBacktester:
    def test_covid_crash_btc(self):
        bt = ScenarioBacktester()
        positions = [{"symbol": "BTCUSDT", "quantity": 1.0, "current_price": 50000, "side": "BUY"}]
        result = bt.run_scenario("covid_crash_2020", positions, 50000)
        assert result.portfolio_impact_pct < 0  # Should lose money
        assert abs(result.portfolio_impact_pct - (-50)) < 1  # ~50% loss for 100% BTC

    def test_short_position_gains(self):
        bt = ScenarioBacktester()
        positions = [{"symbol": "BTCUSDT", "quantity": 1.0, "current_price": 50000, "side": "SELL"}]
        result = bt.run_scenario("covid_crash_2020", positions, 50000)
        assert result.portfolio_impact_pct > 0  # Shorts gain in crash

    def test_all_scenarios_run(self):
        bt = ScenarioBacktester()
        positions = [{"symbol": "BTCUSDT", "quantity": 0.5, "current_price": 50000, "side": "BUY"}]
        results = bt.run_all_scenarios(positions, 50000)
        assert len(results) >= 5

    def test_stress_test(self):
        bt = ScenarioBacktester()
        positions = [{"symbol": "ETHUSDT", "quantity": 10, "current_price": 3000, "side": "BUY"}]
        results = bt.stress_test(positions, 30000)
        assert len(results) >= 5
        # Impact should increase with larger drops
        for i in range(1, len(results)):
            assert results[i]["portfolio_impact_usd"] <= results[i-1]["portfolio_impact_usd"]


class TestSystemicRisk:
    def test_low_risk_uncorrelated(self):
        monitor = SystemicRiskMonitor()
        # Uncorrelated random walks
        rng = np.random.RandomState(42)
        prices = {
            "BTCUSDT": np.cumsum(rng.randn(50)) + 50000,
            "ETHUSDT": np.cumsum(rng.randn(50)) + 3000,
        }
        report = monitor.analyze(prices)
        assert report.risk_score >= 0
        assert report.risk_level in ("low", "elevated", "high", "critical")

    def test_high_risk_correlated(self):
        monitor = SystemicRiskMonitor()
        # Perfectly correlated crash
        base = np.linspace(100, 50, 50)
        prices = {
            "BTCUSDT": base * 500,
            "ETHUSDT": base * 30,
            "SOLUSDT": base * 1.5,
        }
        report = monitor.analyze(prices)
        assert report.correlation_avg > 0.8
        assert report.risk_level in ("high", "critical")


class TestHedgingSuggester:
    def test_concentrated_position(self):
        hs = HedgingSuggester()
        positions = [
            {"symbol": "BTCUSDT", "quantity": 1, "current_price": 50000, "side": "BUY", "unrealized_pnl": 0}
        ]
        suggestions = hs.analyze_and_suggest(positions, 50000, max_concentration_pct=40)
        # 100% in one asset should trigger concentration warning
        assert any(s.action == "REDUCE" for s in suggestions)

    def test_no_suggestions_for_balanced(self):
        hs = HedgingSuggester()
        positions = [
            {"symbol": "BTCUSDT", "quantity": 0.1, "current_price": 50000, "side": "BUY", "unrealized_pnl": 100},
            {"symbol": "ETHUSDT", "quantity": 1.0, "current_price": 3000, "side": "BUY", "unrealized_pnl": 50},
        ]
        suggestions = hs.analyze_and_suggest(positions, 50000, max_concentration_pct=50)
        # Moderate positions shouldn't trigger concentration
        concentration_warnings = [s for s in suggestions if s.action == "REDUCE" and "concentration" in s.reason.lower()]
        assert len(concentration_warnings) == 0

    def test_empty_portfolio(self):
        hs = HedgingSuggester()
        suggestions = hs.analyze_and_suggest([], 10000)
        assert len(suggestions) == 0
