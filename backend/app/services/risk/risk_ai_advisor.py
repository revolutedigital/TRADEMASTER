"""AI-powered risk advisor - intelligent risk analysis and recommendations."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


class RiskSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskInsight:
    """A single risk insight from the AI advisor."""
    severity: RiskSeverity
    category: str
    title: str
    description: str
    recommendation: str
    metric_value: float | None = None
    metric_threshold: float | None = None


class RiskAIAdvisor:
    """
    AI-powered risk advisor that analyzes portfolio risk and provides
    intelligent, contextual recommendations.

    Analyzes:
    - Portfolio composition and concentration
    - Historical performance patterns
    - Current market conditions
    - Risk metric thresholds
    - Correlation dynamics
    - Drawdown patterns
    """

    def __init__(self):
        self._analysis_cache: dict = {}
        logger.info("risk_ai_advisor_initialized")

    def analyze_portfolio(self, portfolio_data: dict) -> dict:
        """
        Comprehensive AI risk analysis of the portfolio.

        Args:
            portfolio_data: {
                'total_equity': float,
                'positions': [{symbol, value, pct, entry_price, current_price, pnl}],
                'daily_returns': [float],  # Historical daily returns
                'current_drawdown': float,
                'max_drawdown': float,
                'sharpe_ratio': float,
                'win_rate': float,
                'avg_win': float,
                'avg_loss': float,
            }
        """
        insights: list[RiskInsight] = []

        # Run all analysis modules
        insights.extend(self._analyze_concentration(portfolio_data))
        insights.extend(self._analyze_drawdown(portfolio_data))
        insights.extend(self._analyze_performance(portfolio_data))
        insights.extend(self._analyze_position_risk(portfolio_data))
        insights.extend(self._analyze_correlation(portfolio_data))
        insights.extend(self._analyze_tail_risk(portfolio_data))

        # Sort by severity
        severity_order = {
            RiskSeverity.CRITICAL: 0,
            RiskSeverity.HIGH: 1,
            RiskSeverity.MEDIUM: 2,
            RiskSeverity.LOW: 3,
            RiskSeverity.INFO: 4,
        }
        insights.sort(key=lambda x: severity_order[x.severity])

        # Generate executive summary
        summary = self._generate_summary(insights, portfolio_data)

        # Risk score (0-100, lower is better)
        risk_score = self._compute_risk_score(insights, portfolio_data)

        return {
            "risk_score": risk_score,
            "risk_level": self._score_to_level(risk_score),
            "summary": summary,
            "insights": [
                {
                    "severity": i.severity.value,
                    "category": i.category,
                    "title": i.title,
                    "description": i.description,
                    "recommendation": i.recommendation,
                    "metric_value": i.metric_value,
                    "metric_threshold": i.metric_threshold,
                }
                for i in insights
            ],
            "n_critical": sum(1 for i in insights if i.severity == RiskSeverity.CRITICAL),
            "n_high": sum(1 for i in insights if i.severity == RiskSeverity.HIGH),
            "n_medium": sum(1 for i in insights if i.severity == RiskSeverity.MEDIUM),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _analyze_concentration(self, data: dict) -> list[RiskInsight]:
        """Analyze portfolio concentration risk."""
        insights = []
        positions = data.get("positions", [])

        if not positions:
            return insights

        # Single position concentration
        for pos in positions:
            pct = pos.get("pct", 0)
            symbol = pos.get("symbol", "UNKNOWN")

            if pct > 0.80:
                insights.append(RiskInsight(
                    severity=RiskSeverity.CRITICAL,
                    category="concentration",
                    title=f"Extreme concentration in {symbol}",
                    description=f"Your portfolio is {pct*100:.0f}% concentrated in {symbol}. "
                               f"A 20% drop would result in a {pct*20:.0f}% portfolio loss.",
                    recommendation=f"Immediately reduce {symbol} to below 50% of portfolio. "
                                  f"Diversify into uncorrelated assets or stablecoins.",
                    metric_value=pct,
                    metric_threshold=0.50,
                ))
            elif pct > 0.50:
                insights.append(RiskInsight(
                    severity=RiskSeverity.HIGH,
                    category="concentration",
                    title=f"High concentration in {symbol}",
                    description=f"{pct*100:.0f}% of your portfolio is in {symbol}. "
                               f"Diversification benefit is limited.",
                    recommendation=f"Consider reducing {symbol} to 30-40% of portfolio.",
                    metric_value=pct,
                    metric_threshold=0.50,
                ))

        # Number of positions
        n_positions = len(positions)
        if n_positions == 1:
            insights.append(RiskInsight(
                severity=RiskSeverity.HIGH,
                category="concentration",
                title="Single-asset portfolio",
                description="Your entire portfolio is in one asset. Zero diversification.",
                recommendation="Add 2-4 uncorrelated assets to reduce idiosyncratic risk.",
                metric_value=1,
                metric_threshold=3,
            ))

        return insights

    def _analyze_drawdown(self, data: dict) -> list[RiskInsight]:
        """Analyze drawdown risk."""
        insights = []
        current_dd = data.get("current_drawdown", 0)
        max_dd = data.get("max_drawdown", 0)

        if current_dd > 0.20:
            insights.append(RiskInsight(
                severity=RiskSeverity.CRITICAL,
                category="drawdown",
                title="Severe drawdown in progress",
                description=f"Portfolio is {current_dd*100:.1f}% below its peak. "
                           f"Recovery requires a {current_dd/(1-current_dd)*100:.1f}% gain.",
                recommendation="Stop all new trades. Consider reducing positions by 50%. "
                              "Focus on capital preservation.",
                metric_value=current_dd,
                metric_threshold=0.20,
            ))
        elif current_dd > 0.10:
            insights.append(RiskInsight(
                severity=RiskSeverity.HIGH,
                category="drawdown",
                title="Significant drawdown",
                description=f"Portfolio is {current_dd*100:.1f}% below peak.",
                recommendation="Reduce position sizes by 30%. Tighten stop-losses. "
                              "Avoid adding to losing positions.",
                metric_value=current_dd,
                metric_threshold=0.10,
            ))

        if max_dd > 0.30:
            insights.append(RiskInsight(
                severity=RiskSeverity.MEDIUM,
                category="drawdown",
                title="Historical max drawdown exceeds 30%",
                description=f"Your worst drawdown was {max_dd*100:.1f}%. This suggests "
                           f"position sizing may be too aggressive.",
                recommendation="Reduce maximum position size. Implement a 15% portfolio "
                              "stop-loss circuit breaker.",
                metric_value=max_dd,
                metric_threshold=0.30,
            ))

        return insights

    def _analyze_performance(self, data: dict) -> list[RiskInsight]:
        """Analyze performance-based risk indicators."""
        insights = []
        returns = data.get("daily_returns", [])
        sharpe = data.get("sharpe_ratio", 0)
        win_rate = data.get("win_rate", 0)
        avg_win = data.get("avg_win", 0)
        avg_loss = data.get("avg_loss", 0)

        if sharpe < 0 and len(returns) > 30:
            insights.append(RiskInsight(
                severity=RiskSeverity.HIGH,
                category="performance",
                title="Negative Sharpe ratio",
                description=f"Sharpe ratio of {sharpe:.2f} indicates returns are not "
                           f"compensating for the risk taken.",
                recommendation="Review strategy effectiveness. Consider reducing leverage "
                              "or switching to a more conservative approach.",
                metric_value=sharpe,
                metric_threshold=0.5,
            ))
        elif 0 < sharpe < 0.5 and len(returns) > 30:
            insights.append(RiskInsight(
                severity=RiskSeverity.MEDIUM,
                category="performance",
                title="Low risk-adjusted returns",
                description=f"Sharpe ratio of {sharpe:.2f} is below the recommended 0.5 minimum.",
                recommendation="Optimize entry/exit timing or reduce position volatility.",
                metric_value=sharpe,
                metric_threshold=0.5,
            ))

        # Win/loss ratio analysis
        if win_rate > 0 and avg_loss > 0:
            profit_factor = (win_rate * avg_win) / ((1 - win_rate) * avg_loss) if avg_loss > 0 else 0
            if profit_factor < 1.0 and profit_factor > 0:
                insights.append(RiskInsight(
                    severity=RiskSeverity.HIGH,
                    category="performance",
                    title="Negative expected value",
                    description=f"Profit factor is {profit_factor:.2f} (below 1.0). "
                               f"Win rate: {win_rate*100:.0f}%, Avg win: ${avg_win:.0f}, "
                               f"Avg loss: ${avg_loss:.0f}.",
                    recommendation="Your strategy loses money on average. Review your "
                                  "edge, tighten stop-losses, or widen profit targets.",
                    metric_value=profit_factor,
                    metric_threshold=1.5,
                ))

        # Consecutive losses detection
        if returns and len(returns) >= 5:
            max_consecutive_losses = 0
            current_streak = 0
            for r in returns[-30:]:
                if r < 0:
                    current_streak += 1
                    max_consecutive_losses = max(max_consecutive_losses, current_streak)
                else:
                    current_streak = 0

            if max_consecutive_losses >= 7:
                insights.append(RiskInsight(
                    severity=RiskSeverity.HIGH,
                    category="performance",
                    title=f"{max_consecutive_losses} consecutive losing days",
                    description="Extended losing streak indicates possible strategy breakdown "
                               "or adverse market regime.",
                    recommendation="Pause trading for 24-48 hours. Review if market regime "
                                  "has changed. Reduce size by 50% when resuming.",
                    metric_value=max_consecutive_losses,
                    metric_threshold=5,
                ))

        return insights

    def _analyze_position_risk(self, data: dict) -> list[RiskInsight]:
        """Analyze individual position risks."""
        insights = []
        positions = data.get("positions", [])

        for pos in positions:
            symbol = pos.get("symbol", "UNKNOWN")
            entry = pos.get("entry_price", 0)
            current = pos.get("current_price", 0)
            pnl_pct = (current / entry - 1) if entry > 0 else 0

            # Large unrealized loss
            if pnl_pct < -0.20:
                insights.append(RiskInsight(
                    severity=RiskSeverity.HIGH,
                    category="position",
                    title=f"Large unrealized loss on {symbol}",
                    description=f"{symbol} is {pnl_pct*100:.1f}% below entry price. "
                               f"Consider whether the original thesis still holds.",
                    recommendation=f"Set a hard stop at -25% if not already. "
                                  f"Avoid averaging down without a clear catalyst.",
                    metric_value=pnl_pct,
                    metric_threshold=-0.10,
                ))

            # Large unrealized gain (risk of giving back profits)
            if pnl_pct > 0.50:
                insights.append(RiskInsight(
                    severity=RiskSeverity.LOW,
                    category="position",
                    title=f"Large unrealized gain on {symbol}",
                    description=f"{symbol} is {pnl_pct*100:.1f}% above entry. "
                               f"Consider securing profits.",
                    recommendation=f"Take partial profit (25-50%) and set a trailing stop "
                                  f"at {current*0.9:.0f} to protect remaining gains.",
                    metric_value=pnl_pct,
                    metric_threshold=0.50,
                ))

        return insights

    def _analyze_correlation(self, data: dict) -> list[RiskInsight]:
        """Analyze correlation risk between positions."""
        insights = []
        positions = data.get("positions", [])

        if len(positions) >= 2:
            # Check if all positions are crypto (high baseline correlation)
            all_crypto = all(
                any(c in pos.get("symbol", "").upper() for c in ["BTC", "ETH", "SOL", "USDT"])
                for pos in positions
            )

            if all_crypto and len(positions) >= 2:
                insights.append(RiskInsight(
                    severity=RiskSeverity.MEDIUM,
                    category="correlation",
                    title="High correlation between positions",
                    description="All positions are cryptocurrency, which typically have "
                               "correlations > 0.7 during market stress. Diversification "
                               "benefit is limited.",
                    recommendation="Consider adding uncorrelated assets: stablecoins "
                                  "for safety, or traditional assets (if available) for "
                                  "true diversification.",
                    metric_value=0.7,
                    metric_threshold=0.5,
                ))

        return insights

    def _analyze_tail_risk(self, data: dict) -> list[RiskInsight]:
        """Analyze tail risk exposure."""
        insights = []
        returns = data.get("daily_returns", [])

        if len(returns) < 30:
            return insights

        returns_array = np.array(returns)

        # Value at Risk (95%)
        var_95 = float(np.percentile(returns_array, 5))
        total_equity = data.get("total_equity", 0)

        if var_95 < -0.05 and total_equity > 0:
            var_dollar = abs(var_95) * total_equity
            insights.append(RiskInsight(
                severity=RiskSeverity.MEDIUM,
                category="tail_risk",
                title="Elevated Value-at-Risk",
                description=f"95% daily VaR is {var_95*100:.1f}% (${var_dollar:,.0f}). "
                           f"On 1 in 20 days, you could lose this much or more.",
                recommendation="Reduce position sizes or add hedges to lower VaR below 3%.",
                metric_value=var_95,
                metric_threshold=-0.03,
            ))

        # Kurtosis (fat tails)
        if len(returns_array) > 10:
            std = np.std(returns_array)
            if std > 0:
                kurtosis = float(np.mean(((returns_array - np.mean(returns_array)) / std) ** 4) - 3)
                if kurtosis > 5:
                    insights.append(RiskInsight(
                        severity=RiskSeverity.MEDIUM,
                        category="tail_risk",
                        title="Fat-tailed return distribution",
                        description=f"Excess kurtosis of {kurtosis:.1f} indicates extreme "
                                   f"moves are more likely than a normal distribution suggests.",
                        recommendation="Use tail-risk-aware position sizing. Consider protective puts "
                                      "for large positions.",
                        metric_value=kurtosis,
                        metric_threshold=3.0,
                    ))

        return insights

    def _generate_summary(self, insights: list[RiskInsight], data: dict) -> str:
        """Generate executive summary of risk analysis."""
        n_critical = sum(1 for i in insights if i.severity == RiskSeverity.CRITICAL)
        n_high = sum(1 for i in insights if i.severity == RiskSeverity.HIGH)
        total_equity = data.get("total_equity", 0)

        if n_critical > 0:
            return (f"URGENT: {n_critical} critical risk issues detected. "
                   f"Portfolio value ${total_equity:,.0f} is at significant risk. "
                   f"Immediate action required to preserve capital.")
        elif n_high > 0:
            return (f"WARNING: {n_high} high-risk issues identified. "
                   f"Review and address within 24 hours to maintain risk targets.")
        elif insights:
            return (f"Portfolio risk is within acceptable parameters. "
                   f"{len(insights)} observations noted for monitoring.")
        else:
            return "No significant risk issues detected. Portfolio is well-balanced."

    def _compute_risk_score(self, insights: list[RiskInsight], data: dict) -> float:
        """Compute overall risk score (0-100, higher = riskier)."""
        score = 0.0

        severity_weights = {
            RiskSeverity.CRITICAL: 25,
            RiskSeverity.HIGH: 15,
            RiskSeverity.MEDIUM: 8,
            RiskSeverity.LOW: 3,
            RiskSeverity.INFO: 1,
        }

        for insight in insights:
            score += severity_weights[insight.severity]

        # Cap at 100
        return min(100.0, round(score, 1))

    def _score_to_level(self, score: float) -> str:
        """Convert risk score to human-readable level."""
        if score >= 75:
            return "critical"
        elif score >= 50:
            return "high"
        elif score >= 25:
            return "moderate"
        elif score >= 10:
            return "low"
        else:
            return "minimal"

    def get_quick_advice(self, question: str, portfolio_data: dict) -> dict:
        """
        Answer a natural language risk question.

        Supports questions like:
        - "What is my biggest risk right now?"
        - "Should I add more BTC?"
        - "How exposed am I to a crash?"
        """
        analysis = self.analyze_portfolio(portfolio_data)
        question_lower = question.lower()

        # Pattern matching for common questions
        if any(w in question_lower for w in ["biggest risk", "main risk", "top risk"]):
            if analysis["insights"]:
                top = analysis["insights"][0]
                answer = f"Your biggest risk is: {top['title']}. {top['description']} " \
                        f"Recommendation: {top['recommendation']}"
            else:
                answer = "No significant risks detected in your current portfolio."

        elif any(w in question_lower for w in ["crash", "black swan", "worst case"]):
            dd = portfolio_data.get("max_drawdown", 0)
            equity = portfolio_data.get("total_equity", 0)
            worst_case = equity * 0.5  # Assume 50% crash scenario
            answer = (f"In a severe crash (50% drawdown), your portfolio could fall to "
                     f"${worst_case:,.0f}. Your historical max drawdown is {dd*100:.1f}%. "
                     f"Consider hedging with protective puts or reducing leverage.")

        elif "add more" in question_lower or "buy more" in question_lower:
            concentration = max(
                (p.get("pct", 0) for p in portfolio_data.get("positions", [])),
                default=0,
            )
            if concentration > 0.5:
                answer = (f"Caution: Your portfolio is already {concentration*100:.0f}% "
                         f"concentrated. Adding more increases single-asset risk. "
                         f"Consider diversifying instead.")
            else:
                answer = (f"Current concentration is {concentration*100:.0f}%, within limits. "
                         f"You could add more, but ensure position doesn't exceed 50% "
                         f"of total portfolio.")

        elif any(w in question_lower for w in ["diversif", "correlation"]):
            n_positions = len(portfolio_data.get("positions", []))
            answer = (f"You have {n_positions} position(s). For crypto portfolios, "
                     f"correlations tend to increase during stress. True diversification "
                     f"requires uncorrelated assets beyond just different cryptocurrencies.")

        else:
            # Generic response with portfolio summary
            answer = (f"Risk score: {analysis['risk_score']}/100 ({analysis['risk_level']}). "
                     f"{analysis['n_critical']} critical, {analysis['n_high']} high, "
                     f"{analysis['n_medium']} medium risk issues. "
                     f"{analysis['summary']}")

        return {
            "question": question,
            "answer": answer,
            "risk_score": analysis["risk_score"],
            "risk_level": analysis["risk_level"],
            "n_insights": len(analysis["insights"]),
            "timestamp": datetime.utcnow().isoformat(),
        }


# Module-level instance
risk_ai_advisor = RiskAIAdvisor()
