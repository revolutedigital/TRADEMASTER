"""Stress testing engine for portfolio risk scenarios."""

from dataclasses import dataclass

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StressTestResult:
    """Result of a single stress test scenario."""

    scenario_name: str
    portfolio_impact: float
    portfolio_value_after: float
    positions_affected: int
    details: dict


SCENARIOS = {
    "flash_crash": {
        "description": "Sudden 30% drop in BTC, 40% in ETH",
        "price_shocks": {"BTCUSDT": -0.30, "ETHUSDT": -0.40},
    },
    "black_swan": {
        "description": "50% drop in BTC, 60% in ETH (2022-style crash)",
        "price_shocks": {"BTCUSDT": -0.50, "ETHUSDT": -0.60},
    },
    "moderate_correction": {
        "description": "10% correction across all assets",
        "price_shocks": {"BTCUSDT": -0.10, "ETHUSDT": -0.10},
    },
    "btc_rally": {
        "description": "20% BTC rally, 30% ETH rally",
        "price_shocks": {"BTCUSDT": 0.20, "ETHUSDT": 0.30},
    },
    "eth_decouple": {
        "description": "BTC flat, ETH drops 25%",
        "price_shocks": {"BTCUSDT": 0.0, "ETHUSDT": -0.25},
    },
}


class StressTestEngine:
    """Runs stress test scenarios against portfolio positions."""

    def run_scenario(
        self,
        scenario_name: str,
        positions: list[dict],
        portfolio_value: float,
    ) -> StressTestResult:
        """Run a single stress test scenario.

        Args:
            scenario_name: Key from SCENARIOS dict
            positions: List of position dicts with symbol, side, quantity, current_price
            portfolio_value: Current total portfolio value
        """
        scenario = SCENARIOS.get(scenario_name)
        if not scenario:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        price_shocks = scenario["price_shocks"]
        total_impact = 0.0
        affected = 0

        for pos in positions:
            symbol = pos.get("symbol", "")
            shock = price_shocks.get(symbol, 0.0)
            if shock == 0.0:
                continue

            affected += 1
            side = pos.get("side", "LONG")
            qty = float(pos.get("quantity", 0))
            price = float(pos.get("current_price", 0))
            notional = qty * price

            if side == "LONG":
                impact = notional * shock
            else:
                impact = notional * (-shock)

            total_impact += impact

        return StressTestResult(
            scenario_name=scenario_name,
            portfolio_impact=round(total_impact, 2),
            portfolio_value_after=round(portfolio_value + total_impact, 2),
            positions_affected=affected,
            details={
                "description": scenario["description"],
                "price_shocks": price_shocks,
                "impact_pct": round(total_impact / portfolio_value * 100, 2) if portfolio_value > 0 else 0,
            },
        )

    def run_all_scenarios(
        self,
        positions: list[dict],
        portfolio_value: float,
    ) -> list[StressTestResult]:
        """Run all predefined stress test scenarios."""
        results = []
        for name in SCENARIOS:
            result = self.run_scenario(name, positions, portfolio_value)
            results.append(result)
        return results


stress_test_engine = StressTestEngine()
