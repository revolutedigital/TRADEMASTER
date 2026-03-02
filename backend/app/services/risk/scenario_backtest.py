"""Risk scenario backtesting - simulate portfolio behavior during historical crises."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ScenarioResult:
    """Result of running a portfolio through a historical scenario."""
    scenario_name: str
    description: str
    initial_value: float
    final_value: float
    min_value: float
    max_drawdown_pct: float
    recovery_days: int | None
    total_return_pct: float
    daily_returns: list[float]
    timeline: list[dict]
    risk_metrics: dict


# Historical crisis scenarios with daily return sequences
HISTORICAL_SCENARIOS = {
    "covid_crash_2020": {
        "name": "COVID-19 Crypto Crash",
        "description": "March 2020: BTC dropped 50% in 2 days as global pandemic fears peaked",
        "start_date": "2020-03-09",
        "end_date": "2020-04-15",
        "btc_daily_returns": [
            -0.02, -0.05, -0.08, -0.15, -0.38, 0.10, -0.05, 0.03,
            0.08, -0.02, 0.05, 0.03, 0.02, -0.01, 0.04, 0.06,
            0.03, 0.02, 0.01, -0.01, 0.03, 0.05, 0.04, 0.02,
            0.01, 0.03, 0.02, 0.01, 0.04, 0.03, 0.02, 0.01,
            0.02, 0.03, 0.04, 0.02, 0.01,
        ],
        "eth_correlation": 0.92,
        "severity": 10,
    },
    "luna_crash_2022": {
        "name": "Terra/LUNA Collapse",
        "description": "May 2022: UST depeg triggered LUNA hyperinflation, destroying $40B in value",
        "start_date": "2022-05-07",
        "end_date": "2022-06-18",
        "btc_daily_returns": [
            -0.03, -0.05, -0.08, -0.12, -0.15, -0.05, 0.03, -0.08,
            -0.10, 0.02, -0.03, 0.01, -0.02, -0.04, 0.01, -0.01,
            -0.03, -0.05, -0.08, -0.15, 0.05, -0.03, -0.02, 0.01,
            -0.01, 0.02, 0.01, -0.01, 0.02, 0.03, 0.01, -0.02,
            0.01, 0.02, 0.01, -0.01, 0.02, 0.03, 0.01, 0.02,
        ],
        "eth_correlation": 0.88,
        "severity": 9,
    },
    "ftx_collapse_2022": {
        "name": "FTX Exchange Collapse",
        "description": "Nov 2022: FTX bankruptcy triggered market-wide panic and contagion fears",
        "start_date": "2022-11-06",
        "end_date": "2022-12-31",
        "btc_daily_returns": [
            -0.02, -0.04, -0.08, -0.13, -0.10, 0.02, -0.05, 0.01,
            -0.03, -0.02, 0.01, -0.01, -0.02, 0.01, 0.02, -0.01,
            0.01, -0.01, 0.02, -0.01, 0.01, 0.02, -0.02, 0.01,
            0.01, 0.02, -0.01, 0.01, -0.01, 0.02, 0.01, -0.01,
            0.01, 0.02, 0.01, -0.01, 0.01, 0.01, -0.01, 0.02,
            0.01, 0.01, -0.01, 0.02, 0.01, -0.01, 0.01, 0.02,
            0.01, 0.01, 0.02, 0.01, -0.01, 0.02,
        ],
        "eth_correlation": 0.90,
        "severity": 8,
    },
    "china_ban_2021": {
        "name": "China Mining Ban",
        "description": "May-June 2021: China banned crypto mining, BTC hashrate dropped 50%",
        "start_date": "2021-05-12",
        "end_date": "2021-07-20",
        "btc_daily_returns": [
            -0.05, -0.10, -0.12, 0.05, -0.08, -0.05, 0.03, -0.03,
            -0.07, -0.10, 0.08, -0.05, 0.02, -0.03, -0.02, 0.01,
            -0.04, -0.08, 0.05, -0.02, 0.01, -0.01, 0.02, -0.02,
            0.03, -0.01, 0.01, -0.02, 0.02, 0.01, -0.01, 0.03,
            0.02, -0.01, 0.01, 0.02, -0.01, 0.03, 0.02, 0.01,
            0.02, 0.01, 0.03, 0.02, 0.01, 0.02, 0.01, 0.03,
            0.02, 0.04, 0.03, 0.02, 0.04, 0.05, 0.03, 0.02,
            0.01, 0.03, 0.04, 0.02, 0.05, 0.03, 0.02, 0.04,
            0.03, 0.02, 0.01, 0.03,
        ],
        "eth_correlation": 0.85,
        "severity": 7,
    },
    "flash_crash_generic": {
        "name": "Flash Crash",
        "description": "Hypothetical flash crash: 30% drop in minutes, rapid partial recovery",
        "start_date": "hypothetical",
        "end_date": "hypothetical",
        "btc_daily_returns": [
            -0.30, 0.15, 0.05, 0.03, 0.02, -0.01, 0.01,
        ],
        "eth_correlation": 0.95,
        "severity": 8,
    },
    "prolonged_bear_2022": {
        "name": "2022 Crypto Winter",
        "description": "Full year 2022 bear market: BTC fell from $47K to $16K (-65%)",
        "start_date": "2022-01-01",
        "end_date": "2022-12-31",
        "btc_daily_returns": [
            # Simplified monthly average daily returns
            -0.005, -0.005, -0.004, -0.003, -0.005, -0.004,
            -0.008, -0.008, -0.010, -0.012, -0.015, -0.008,
            0.005, -0.003, -0.005, -0.008, -0.012, -0.015,
            -0.005, 0.003, -0.002, -0.003, 0.002, -0.001,
            -0.003, -0.002, 0.001, -0.001, 0.002, 0.001,
        ],
        "eth_correlation": 0.90,
        "severity": 7,
    },
}


class ScenarioBacktester:
    """
    Replay historical crisis scenarios against current portfolio.

    Features:
    - Replay portfolio through actual historical crash sequences
    - Multi-asset correlation modeling
    - What-if analysis with different position sizes
    - Recovery time estimation
    - Comparative analysis across scenarios
    """

    def __init__(self):
        self.scenarios = HISTORICAL_SCENARIOS
        logger.info("scenario_backtester_initialized", n_scenarios=len(self.scenarios))

    def run_scenario(self, scenario_name: str, portfolio: dict,
                     apply_risk_rules: bool = True) -> ScenarioResult:
        """
        Run portfolio through a historical scenario.

        Args:
            scenario_name: Key from HISTORICAL_SCENARIOS
            portfolio: dict with 'total_value', 'positions' ({symbol: {value, pct}})
            apply_risk_rules: Whether to apply stop-loss and risk management
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}. "
                           f"Available: {list(self.scenarios.keys())}")

        scenario = self.scenarios[scenario_name]
        btc_returns = scenario["btc_daily_returns"]
        eth_correlation = scenario["eth_correlation"]
        total_value = portfolio.get("total_value", 10000.0)
        positions = portfolio.get("positions", {})

        # Generate correlated returns for each position
        rng = np.random.RandomState(42)
        daily_portfolio_returns = []
        timeline = []
        current_value = total_value
        peak_value = total_value
        min_value = total_value
        max_drawdown = 0.0

        # Position allocations
        btc_pct = 0.0
        eth_pct = 0.0
        cash_pct = 0.0
        for symbol, pos_info in positions.items():
            pct = pos_info.get("pct", 0) if isinstance(pos_info, dict) else float(pos_info)
            if "BTC" in symbol.upper():
                btc_pct = pct
            elif "ETH" in symbol.upper():
                eth_pct = pct

        cash_pct = 1.0 - btc_pct - eth_pct
        if cash_pct < 0:
            cash_pct = 0

        stop_loss_triggered = False
        stop_loss_day = None

        for day, btc_ret in enumerate(btc_returns):
            # Generate ETH return (correlated)
            noise = rng.normal(0, 0.02)
            eth_ret = btc_ret * eth_correlation + noise * (1 - eth_correlation)

            # Portfolio return
            if stop_loss_triggered and apply_risk_rules:
                port_ret = 0.0  # Flat after stop-loss
            else:
                port_ret = btc_pct * btc_ret + eth_pct * eth_ret
                # Cash earns nothing during crash

            current_value *= (1 + port_ret)
            daily_portfolio_returns.append(port_ret)

            peak_value = max(peak_value, current_value)
            min_value = min(min_value, current_value)
            drawdown = (peak_value - current_value) / peak_value
            max_drawdown = max(max_drawdown, drawdown)

            # Stop-loss check (10% drawdown)
            if apply_risk_rules and drawdown > 0.10 and not stop_loss_triggered:
                stop_loss_triggered = True
                stop_loss_day = day
                # Sell everything at current price
                current_value = current_value  # Already reflected in the return

            timeline.append({
                "day": day,
                "btc_return": round(btc_ret, 4),
                "eth_return": round(eth_ret, 4),
                "portfolio_return": round(port_ret, 4),
                "portfolio_value": round(current_value, 2),
                "drawdown_pct": round(drawdown * 100, 2),
                "stop_loss_active": stop_loss_triggered,
            })

        # Recovery analysis
        recovery_days = None
        if current_value < total_value:
            # Estimate recovery based on average post-crisis returns
            avg_recovery_rate = 0.003  # ~0.3% daily
            deficit = total_value - current_value
            if avg_recovery_rate > 0 and current_value > 0:
                days_needed = deficit / (current_value * avg_recovery_rate)
                recovery_days = int(days_needed)
        else:
            recovery_days = 0  # Already recovered

        # Risk metrics
        returns_array = np.array(daily_portfolio_returns)
        risk_metrics = {
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "total_return_pct": round((current_value / total_value - 1) * 100, 2),
            "worst_day_pct": round(float(np.min(returns_array)) * 100, 2) if len(returns_array) > 0 else 0,
            "best_day_pct": round(float(np.max(returns_array)) * 100, 2) if len(returns_array) > 0 else 0,
            "volatility_annualized": round(float(np.std(returns_array) * np.sqrt(365)) * 100, 2) if len(returns_array) > 0 else 0,
            "sharpe_ratio": round(
                float(np.mean(returns_array) / (np.std(returns_array) + 1e-10) * np.sqrt(365)), 2
            ) if len(returns_array) > 0 else 0,
            "stop_loss_triggered": stop_loss_triggered,
            "stop_loss_day": stop_loss_day,
            "calmar_ratio": round(
                float((current_value / total_value - 1) / (max_drawdown + 1e-10)), 2
            ),
            "days_underwater": sum(1 for t in timeline if t["drawdown_pct"] > 0),
        }

        return ScenarioResult(
            scenario_name=scenario_name,
            description=scenario["description"],
            initial_value=total_value,
            final_value=round(current_value, 2),
            min_value=round(min_value, 2),
            max_drawdown_pct=round(max_drawdown * 100, 2),
            recovery_days=recovery_days,
            total_return_pct=round((current_value / total_value - 1) * 100, 2),
            daily_returns=[round(r, 4) for r in daily_portfolio_returns],
            timeline=timeline,
            risk_metrics=risk_metrics,
        )

    def run_all_scenarios(self, portfolio: dict,
                          apply_risk_rules: bool = True) -> dict:
        """Run portfolio through ALL historical scenarios."""
        results = {}
        for name in self.scenarios:
            try:
                result = self.run_scenario(name, portfolio, apply_risk_rules)
                results[name] = {
                    "scenario": result.scenario_name,
                    "description": result.description,
                    "initial_value": result.initial_value,
                    "final_value": result.final_value,
                    "min_value": result.min_value,
                    "max_drawdown_pct": result.max_drawdown_pct,
                    "total_return_pct": result.total_return_pct,
                    "recovery_days": result.recovery_days,
                    "risk_metrics": result.risk_metrics,
                }
            except Exception as e:
                logger.warning("scenario_failed", scenario=name, error=str(e))
                results[name] = {"error": str(e)}

        # Summary statistics
        valid = [r for r in results.values() if "error" not in r]
        summary = {
            "worst_scenario": min(valid, key=lambda x: x["total_return_pct"])["scenario"] if valid else None,
            "best_scenario": max(valid, key=lambda x: x["total_return_pct"])["scenario"] if valid else None,
            "avg_max_drawdown": round(np.mean([r["max_drawdown_pct"] for r in valid]), 2) if valid else 0,
            "avg_return": round(np.mean([r["total_return_pct"] for r in valid]), 2) if valid else 0,
            "scenarios_with_stop_loss": sum(
                1 for r in valid if r["risk_metrics"].get("stop_loss_triggered", False)
            ),
        }

        return {
            "portfolio": portfolio,
            "results": results,
            "summary": summary,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def what_if_analysis(self, portfolio: dict, scenario_name: str,
                         position_adjustments: list[dict]) -> list[dict]:
        """
        What-if analysis: compare different position sizes in same scenario.

        Args:
            position_adjustments: list of dicts with 'label' and 'positions'
        """
        results = []
        for adj in position_adjustments:
            modified_portfolio = {
                "total_value": portfolio["total_value"],
                "positions": adj["positions"],
            }
            result = self.run_scenario(scenario_name, modified_portfolio)
            results.append({
                "label": adj.get("label", "Variant"),
                "positions": adj["positions"],
                "final_value": result.final_value,
                "max_drawdown_pct": result.max_drawdown_pct,
                "total_return_pct": result.total_return_pct,
                "risk_metrics": result.risk_metrics,
            })

        return results

    def get_scenario_catalog(self) -> list[dict]:
        """Get catalog of available scenarios."""
        return [
            {
                "id": key,
                "name": scenario["name"],
                "description": scenario["description"],
                "start_date": scenario["start_date"],
                "end_date": scenario["end_date"],
                "severity": scenario["severity"],
                "duration_days": len(scenario["btc_daily_returns"]),
            }
            for key, scenario in self.scenarios.items()
        ]


# Module-level instance
scenario_backtester = ScenarioBacktester()
