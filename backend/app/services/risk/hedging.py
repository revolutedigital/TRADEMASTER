"""Dynamic hedging suggestions based on portfolio exposure."""
from dataclasses import dataclass

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HedgeSuggestion:
    action: str  # "REDUCE", "HEDGE", "DIVERSIFY", "REBALANCE"
    symbol: str
    size_pct: float  # Suggested size as % of position
    reason: str
    urgency: str  # "low", "medium", "high"
    estimated_risk_reduction_pct: float


class HedgingSuggester:
    """Suggest hedging actions based on portfolio analysis."""

    def analyze_and_suggest(
        self,
        positions: list[dict],  # [{symbol, quantity, current_price, side, unrealized_pnl}]
        total_equity: float,
        max_concentration_pct: float = 40.0,
        max_drawdown_tolerance_pct: float = 15.0,
    ) -> list[HedgeSuggestion]:
        """Analyze portfolio and suggest hedging actions."""
        suggestions = []

        if not positions or total_equity <= 0:
            return suggestions

        # Calculate position sizes
        position_data = []
        total_long = 0.0
        total_short = 0.0

        for pos in positions:
            value = float(pos.get("quantity", 0)) * float(pos.get("current_price", 0))
            pct = (value / total_equity * 100) if total_equity > 0 else 0
            side = pos.get("side", "BUY")

            if side == "BUY":
                total_long += value
            else:
                total_short += value

            position_data.append({
                **pos,
                "value": value,
                "pct_of_equity": pct,
            })

        net_exposure = total_long - total_short
        net_exposure_pct = (net_exposure / total_equity * 100) if total_equity > 0 else 0
        gross_exposure = total_long + total_short
        gross_exposure_pct = (gross_exposure / total_equity * 100) if total_equity > 0 else 0

        # 1. Concentration risk
        for pd in position_data:
            if pd["pct_of_equity"] > max_concentration_pct:
                suggestions.append(HedgeSuggestion(
                    action="REDUCE",
                    symbol=pd["symbol"],
                    size_pct=round(pd["pct_of_equity"] - max_concentration_pct, 1),
                    reason=f"Position is {pd['pct_of_equity']:.1f}% of portfolio (limit: {max_concentration_pct}%)",
                    urgency="high",
                    estimated_risk_reduction_pct=round((pd["pct_of_equity"] - max_concentration_pct) * 0.7, 1),
                ))

        # 2. Directional bias
        if abs(net_exposure_pct) > 80:
            direction = "long" if net_exposure_pct > 0 else "short"
            suggestions.append(HedgeSuggestion(
                action="HEDGE",
                symbol="portfolio",
                size_pct=round(abs(net_exposure_pct) - 60, 1),
                reason=f"Portfolio is {abs(net_exposure_pct):.0f}% net {direction} - consider hedging",
                urgency="medium",
                estimated_risk_reduction_pct=round((abs(net_exposure_pct) - 60) * 0.5, 1),
            ))

        # 3. Gross exposure too high
        if gross_exposure_pct > 150:
            suggestions.append(HedgeSuggestion(
                action="REDUCE",
                symbol="portfolio",
                size_pct=round(gross_exposure_pct - 100, 1),
                reason=f"Gross exposure at {gross_exposure_pct:.0f}% - overleveraged",
                urgency="high",
                estimated_risk_reduction_pct=round((gross_exposure_pct - 100) * 0.6, 1),
            ))

        # 4. Losing positions (cut losers)
        for pd in position_data:
            unrealized = float(pd.get("unrealized_pnl", 0))
            if unrealized < 0 and abs(unrealized) > pd["value"] * 0.10:
                loss_pct = abs(unrealized) / pd["value"] * 100 if pd["value"] > 0 else 0
                suggestions.append(HedgeSuggestion(
                    action="REDUCE",
                    symbol=pd["symbol"],
                    size_pct=50.0,
                    reason=f"Position down {loss_pct:.1f}% - consider cutting losses",
                    urgency="medium" if loss_pct < 15 else "high",
                    estimated_risk_reduction_pct=round(loss_pct * 0.5, 1),
                ))

        # 5. Diversification
        symbols = set(pd["symbol"] for pd in position_data)
        asset_types = set()
        for s in symbols:
            if "BTC" in s.upper():
                asset_types.add("btc")
            elif "ETH" in s.upper():
                asset_types.add("eth")
            else:
                asset_types.add("alt")

        if len(asset_types) <= 1 and len(positions) > 1:
            suggestions.append(HedgeSuggestion(
                action="DIVERSIFY",
                symbol="portfolio",
                size_pct=20.0,
                reason=f"All positions in {list(asset_types)[0].upper()} - no diversification",
                urgency="low",
                estimated_risk_reduction_pct=15.0,
            ))

        return sorted(suggestions, key=lambda s: {"high": 0, "medium": 1, "low": 2}[s.urgency])


hedging_suggester = HedgingSuggester()
