"""Scenario backtesting: replay historical crisis events against current portfolio."""
from dataclasses import dataclass

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ScenarioResult:
    scenario_name: str
    description: str
    portfolio_impact_pct: float  # How much portfolio would drop
    portfolio_impact_usd: float
    max_drawdown_pct: float
    recovery_candles: int | None  # Estimated candles to recover (None if unrecoverable)
    assets_most_affected: list[dict]


# Historical crisis scenarios with approximate return distributions
CRISIS_SCENARIOS = {
    "covid_crash_2020": {
        "description": "COVID-19 crash (March 2020): BTC dropped ~50% in 2 days",
        "btc_return": -0.50,
        "eth_return": -0.55,
        "alt_return": -0.65,
        "duration_days": 2,
        "recovery_days": 55,
    },
    "luna_collapse_2022": {
        "description": "LUNA/UST collapse (May 2022): Cascading liquidations",
        "btc_return": -0.30,
        "eth_return": -0.35,
        "alt_return": -0.70,
        "duration_days": 7,
        "recovery_days": None,  # Never fully recovered
    },
    "ftx_collapse_2022": {
        "description": "FTX collapse (Nov 2022): Exchange failure & contagion",
        "btc_return": -0.25,
        "eth_return": -0.28,
        "alt_return": -0.50,
        "duration_days": 5,
        "recovery_days": 60,
    },
    "china_ban_2021": {
        "description": "China mining ban (May 2021): Hash rate crash",
        "btc_return": -0.35,
        "eth_return": -0.40,
        "alt_return": -0.55,
        "duration_days": 14,
        "recovery_days": 90,
    },
    "flash_crash_10pct": {
        "description": "Generic 10% flash crash: Sudden market-wide drop",
        "btc_return": -0.10,
        "eth_return": -0.12,
        "alt_return": -0.18,
        "duration_days": 1,
        "recovery_days": 5,
    },
    "black_swan_30pct": {
        "description": "Black swan event: 30% overnight crash",
        "btc_return": -0.30,
        "eth_return": -0.35,
        "alt_return": -0.50,
        "duration_days": 1,
        "recovery_days": 45,
    },
}


class ScenarioBacktester:
    """Replay historical crisis scenarios against a portfolio."""

    def _classify_asset(self, symbol: str) -> str:
        """Classify asset for scenario mapping."""
        s = symbol.upper()
        if "BTC" in s:
            return "btc"
        elif "ETH" in s:
            return "eth"
        else:
            return "alt"

    def run_scenario(
        self,
        scenario_name: str,
        positions: list[dict],  # [{symbol, quantity, current_price, side}]
        total_equity: float,
    ) -> ScenarioResult:
        """Run a single crisis scenario against current positions."""
        if scenario_name not in CRISIS_SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        scenario = CRISIS_SCENARIOS[scenario_name]
        total_impact = 0.0
        assets_affected = []

        for pos in positions:
            symbol = pos["symbol"]
            quantity = float(pos.get("quantity", 0))
            price = float(pos.get("current_price", 0))
            side = pos.get("side", "BUY")

            asset_type = self._classify_asset(symbol)
            scenario_return = scenario.get(f"{asset_type}_return", scenario.get("alt_return", -0.30))

            position_value = quantity * price
            # Long positions lose when market drops, shorts gain
            if side == "BUY":
                impact = position_value * scenario_return
            else:
                impact = position_value * (-scenario_return)

            total_impact += impact
            assets_affected.append({
                "symbol": symbol,
                "position_value": round(position_value, 2),
                "scenario_return_pct": round(scenario_return * 100, 1),
                "impact_usd": round(impact, 2),
            })

        impact_pct = (total_impact / total_equity * 100) if total_equity > 0 else 0

        return ScenarioResult(
            scenario_name=scenario_name,
            description=scenario["description"],
            portfolio_impact_pct=round(impact_pct, 2),
            portfolio_impact_usd=round(total_impact, 2),
            max_drawdown_pct=round(abs(impact_pct), 2),
            recovery_candles=scenario.get("recovery_days"),
            assets_most_affected=sorted(assets_affected, key=lambda x: x["impact_usd"]),
        )

    def run_all_scenarios(
        self,
        positions: list[dict],
        total_equity: float,
    ) -> list[ScenarioResult]:
        """Run all crisis scenarios."""
        results = []
        for name in CRISIS_SCENARIOS:
            result = self.run_scenario(name, positions, total_equity)
            results.append(result)
        return sorted(results, key=lambda r: r.portfolio_impact_pct)

    def stress_test(
        self,
        positions: list[dict],
        total_equity: float,
        drops: list[float] | None = None,
    ) -> list[dict]:
        """Simple stress test: what happens at various market drop levels."""
        if drops is None:
            drops = [-0.05, -0.10, -0.15, -0.20, -0.30, -0.50]

        results = []
        for drop in drops:
            total_impact = 0.0
            for pos in positions:
                qty = float(pos.get("quantity", 0))
                price = float(pos.get("current_price", 0))
                side = pos.get("side", "BUY")
                impact = qty * price * drop if side == "BUY" else qty * price * (-drop)
                total_impact += impact

            impact_pct = (total_impact / total_equity * 100) if total_equity > 0 else 0
            remaining = total_equity + total_impact

            results.append({
                "market_drop_pct": round(drop * 100, 1),
                "portfolio_impact_usd": round(total_impact, 2),
                "portfolio_impact_pct": round(impact_pct, 2),
                "remaining_equity": round(remaining, 2),
                "margin_call_risk": remaining < total_equity * 0.3,
            })

        return results


scenario_backtester = ScenarioBacktester()
