"""Tail risk hedging suggestions for portfolio protection."""
from dataclasses import dataclass

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HedgeSuggestion:
    action: str  # REDUCE, HEDGE, REBALANCE
    symbol: str
    size_pct: float
    reason: str
    urgency: str  # low, medium, high


class HedgingSuggester:
    """Analyze portfolio exposure and suggest hedging actions."""

    def suggest(self, positions: list[dict], risk_metrics: dict) -> list[HedgeSuggestion]:
        """Generate hedging suggestions based on current portfolio state.

        Args:
            positions: List of {symbol, side, weight, unrealized_pnl} dicts
            risk_metrics: Dict with drawdown, var, exposure data
        """
        suggestions = []

        total_exposure = sum(abs(p.get("weight", 0)) for p in positions)
        daily_drawdown = risk_metrics.get("daily_drawdown", 0)

        # Check concentration risk
        for pos in positions:
            weight = abs(pos.get("weight", 0))
            if weight > 0.5:
                suggestions.append(HedgeSuggestion(
                    action="REDUCE",
                    symbol=pos["symbol"],
                    size_pct=round((weight - 0.3) * 100, 1),
                    reason=f"Concentration risk: {pos['symbol']} is {weight:.0%} of portfolio",
                    urgency="high",
                ))

        # Check overall exposure
        if total_exposure > 0.8:
            suggestions.append(HedgeSuggestion(
                action="REDUCE",
                symbol="PORTFOLIO",
                size_pct=round((total_exposure - 0.6) * 100, 1),
                reason=f"Total exposure {total_exposure:.0%} exceeds 80% threshold",
                urgency="high",
            ))

        # Check drawdown
        if daily_drawdown < -0.02:
            suggestions.append(HedgeSuggestion(
                action="REDUCE",
                symbol="PORTFOLIO",
                size_pct=30,
                reason=f"Daily drawdown {daily_drawdown:.2%} approaching circuit breaker",
                urgency="high",
            ))
        elif daily_drawdown < -0.01:
            suggestions.append(HedgeSuggestion(
                action="REBALANCE",
                symbol="PORTFOLIO",
                size_pct=15,
                reason=f"Daily drawdown {daily_drawdown:.2%} - consider reducing exposure",
                urgency="medium",
            ))

        # All long positions - suggest diversification
        all_long = all(p.get("side") == "LONG" for p in positions)
        if all_long and len(positions) > 0:
            suggestions.append(HedgeSuggestion(
                action="HEDGE",
                symbol="PORTFOLIO",
                size_pct=10,
                reason="Portfolio is 100% long - no directional hedging",
                urgency="low",
            ))

        return suggestions


hedging_suggester = HedgingSuggester()
