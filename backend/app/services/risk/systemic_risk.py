"""Systemic risk monitoring - cross-market correlation and contagion detection."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SystemicRiskLevel:
    """Systemic risk assessment."""
    level: str  # low, moderate, elevated, high, critical
    score: float  # 0-100
    indicators: dict
    recommendations: list[str]
    timestamp: str


class SystemicRiskMonitor:
    """
    Monitor systemic risk across cryptocurrency and traditional markets.

    Tracks:
    - Cross-asset correlations (BTC/ETH, crypto/equities, crypto/gold)
    - Volatility regime (VIX-equivalent for crypto)
    - Liquidity conditions
    - Contagion indicators
    - Market microstructure stress
    """

    def __init__(self):
        self._price_history: dict[str, list[float]] = {}
        self._correlation_history: list[dict] = []
        self._risk_scores: list[float] = []

        # Reference assets for cross-market analysis
        self.tracked_assets = {
            "crypto": ["BTCUSDT", "ETHUSDT"],
            "traditional": ["SPX", "GOLD", "DXY", "VIX"],
            "stablecoins": ["USDTUSDT", "USDCUSDT"],
        }

        logger.info("systemic_risk_monitor_initialized")

    def update_prices(self, symbol: str, price: float) -> None:
        """Update price for an asset."""
        if symbol not in self._price_history:
            self._price_history[symbol] = []
        self._price_history[symbol].append(price)
        # Keep last 500 data points
        if len(self._price_history[symbol]) > 500:
            self._price_history[symbol] = self._price_history[symbol][-500:]

    def assess_systemic_risk(self) -> SystemicRiskLevel:
        """Run comprehensive systemic risk assessment."""
        indicators = {}
        risk_scores = []

        # 1. Cross-asset correlation analysis
        corr_score, corr_details = self._analyze_correlations()
        indicators["correlation"] = corr_details
        risk_scores.append(corr_score)

        # 2. Volatility regime
        vol_score, vol_details = self._analyze_volatility()
        indicators["volatility"] = vol_details
        risk_scores.append(vol_score)

        # 3. Liquidity stress
        liq_score, liq_details = self._analyze_liquidity()
        indicators["liquidity"] = liq_details
        risk_scores.append(liq_score)

        # 4. Contagion risk
        cont_score, cont_details = self._analyze_contagion()
        indicators["contagion"] = cont_details
        risk_scores.append(cont_score)

        # 5. Market microstructure
        micro_score, micro_details = self._analyze_microstructure()
        indicators["microstructure"] = micro_details
        risk_scores.append(micro_score)

        # Aggregate score (weighted)
        weights = [0.25, 0.25, 0.20, 0.20, 0.10]
        overall_score = sum(s * w for s, w in zip(risk_scores, weights))

        # Determine level
        if overall_score >= 80:
            level = "critical"
        elif overall_score >= 60:
            level = "high"
        elif overall_score >= 40:
            level = "elevated"
        elif overall_score >= 20:
            level = "moderate"
        else:
            level = "low"

        recommendations = self._generate_recommendations(level, indicators)

        result = SystemicRiskLevel(
            level=level,
            score=round(overall_score, 1),
            indicators=indicators,
            recommendations=recommendations,
            timestamp=datetime.utcnow().isoformat(),
        )

        self._risk_scores.append(overall_score)
        logger.info("systemic_risk_assessed", level=level, score=round(overall_score, 1))

        return result

    def _analyze_correlations(self) -> tuple[float, dict]:
        """Analyze cross-asset correlations. High correlation = high systemic risk."""
        correlations = {}
        risk_score = 0.0

        symbols = [s for s in self._price_history if len(self._price_history[s]) >= 20]

        if len(symbols) < 2:
            return 0.0, {"status": "insufficient_data", "n_assets": len(symbols)}

        # Compute pairwise correlations
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                s1, s2 = symbols[i], symbols[j]
                p1 = np.array(self._price_history[s1][-100:])
                p2 = np.array(self._price_history[s2][-100:])
                min_len = min(len(p1), len(p2))
                if min_len < 10:
                    continue

                r1 = np.diff(p1[-min_len:]) / p1[-min_len:-1]
                r2 = np.diff(p2[-min_len:]) / p2[-min_len:-1]

                corr = float(np.corrcoef(r1, r2)[0, 1])
                correlations[f"{s1}/{s2}"] = round(corr, 4)

                # High positive correlation during stress = contagion risk
                if abs(corr) > 0.8:
                    risk_score += 20
                elif abs(corr) > 0.6:
                    risk_score += 10

        avg_corr = np.mean(list(correlations.values())) if correlations else 0
        risk_score = min(100, risk_score + abs(avg_corr) * 30)

        return risk_score, {
            "pairwise": correlations,
            "average_correlation": round(float(avg_corr), 4),
            "n_high_correlations": sum(1 for c in correlations.values() if abs(c) > 0.7),
        }

    def _analyze_volatility(self) -> tuple[float, dict]:
        """Analyze volatility regime across assets."""
        volatilities = {}

        for symbol, prices in self._price_history.items():
            if len(prices) < 20:
                continue
            returns = np.diff(prices[-50:]) / np.array(prices[-50:-1])
            vol = float(np.std(returns) * np.sqrt(365))  # Annualized
            volatilities[symbol] = round(vol, 4)

        if not volatilities:
            return 0.0, {"status": "insufficient_data"}

        avg_vol = np.mean(list(volatilities.values()))
        max_vol = max(volatilities.values())

        # Risk score based on volatility level
        if avg_vol > 1.5:  # >150% annualized
            risk_score = 90
        elif avg_vol > 1.0:
            risk_score = 70
        elif avg_vol > 0.6:
            risk_score = 40
        elif avg_vol > 0.3:
            risk_score = 20
        else:
            risk_score = 5

        return risk_score, {
            "per_asset": volatilities,
            "average_annualized": round(float(avg_vol), 4),
            "max_annualized": round(float(max_vol), 4),
            "regime": "extreme" if avg_vol > 1.0 else "high" if avg_vol > 0.6 else "normal",
        }

    def _analyze_liquidity(self) -> tuple[float, dict]:
        """Analyze liquidity conditions."""
        # Proxy: use volume changes and price impact
        liquidity_scores = {}

        for symbol, prices in self._price_history.items():
            if len(prices) < 30:
                continue

            recent = prices[-10:]
            older = prices[-30:-10]

            # Price stability as liquidity proxy
            recent_vol = float(np.std(np.diff(recent) / np.array(recent[:-1]))) if len(recent) > 1 else 0
            older_vol = float(np.std(np.diff(older) / np.array(older[:-1]))) if len(older) > 1 else 0

            if older_vol > 0:
                vol_change = recent_vol / older_vol
            else:
                vol_change = 1.0

            liquidity_scores[symbol] = round(vol_change, 4)

        if not liquidity_scores:
            return 0.0, {"status": "insufficient_data"}

        avg_change = np.mean(list(liquidity_scores.values()))

        # Rising volatility = deteriorating liquidity
        if avg_change > 3.0:
            risk_score = 90
        elif avg_change > 2.0:
            risk_score = 60
        elif avg_change > 1.5:
            risk_score = 30
        else:
            risk_score = 10

        return risk_score, {
            "volatility_ratio": {k: v for k, v in liquidity_scores.items()},
            "average_ratio": round(float(avg_change), 4),
            "condition": "stressed" if avg_change > 2 else "tightening" if avg_change > 1.5 else "normal",
        }

    def _analyze_contagion(self) -> tuple[float, dict]:
        """Detect potential contagion between markets."""
        # Look for cascading price drops
        cascade_count = 0
        affected_assets = []

        for symbol, prices in self._price_history.items():
            if len(prices) < 5:
                continue
            recent_return = (prices[-1] / prices[-5] - 1) if prices[-5] > 0 else 0
            if recent_return < -0.10:  # >10% drop in 5 periods
                cascade_count += 1
                affected_assets.append({
                    "symbol": symbol,
                    "return_5d": round(recent_return, 4),
                })

        total_assets = len([s for s in self._price_history if len(self._price_history[s]) >= 5])
        contagion_ratio = cascade_count / max(total_assets, 1)

        if contagion_ratio > 0.7:
            risk_score = 95
        elif contagion_ratio > 0.5:
            risk_score = 70
        elif contagion_ratio > 0.3:
            risk_score = 40
        else:
            risk_score = 10

        return risk_score, {
            "affected_assets": affected_assets,
            "contagion_ratio": round(contagion_ratio, 4),
            "n_declining": cascade_count,
            "total_monitored": total_assets,
        }

    def _analyze_microstructure(self) -> tuple[float, dict]:
        """Analyze market microstructure stress indicators."""
        stress_indicators = {}

        for symbol, prices in self._price_history.items():
            if len(prices) < 20:
                continue

            returns = np.diff(prices[-20:]) / np.array(prices[-20:-1])

            # Kurtosis (excess) - fat tails indicate stress
            if len(returns) > 3:
                kurtosis = float(np.mean((returns - np.mean(returns)) ** 4) / (np.std(returns) ** 4 + 1e-10) - 3)
            else:
                kurtosis = 0.0

            # Serial correlation - mean reversion or momentum
            if len(returns) > 1:
                autocorr = float(np.corrcoef(returns[:-1], returns[1:])[0, 1])
            else:
                autocorr = 0.0

            stress_indicators[symbol] = {
                "excess_kurtosis": round(kurtosis, 4),
                "autocorrelation": round(autocorr, 4),
            }

        if not stress_indicators:
            return 0.0, {"status": "insufficient_data"}

        avg_kurtosis = np.mean([v["excess_kurtosis"] for v in stress_indicators.values()])
        avg_autocorr = np.mean([abs(v["autocorrelation"]) for v in stress_indicators.values()])

        risk_score = min(100, max(0, avg_kurtosis * 5 + avg_autocorr * 50))

        return risk_score, {
            "per_asset": stress_indicators,
            "avg_excess_kurtosis": round(float(avg_kurtosis), 4),
            "avg_abs_autocorrelation": round(float(avg_autocorr), 4),
        }

    def _generate_recommendations(self, level: str, indicators: dict) -> list[str]:
        """Generate risk management recommendations."""
        recs = []

        if level in ("critical", "high"):
            recs.append("URGENT: Reduce total portfolio exposure by 50%")
            recs.append("Activate circuit breaker - pause automated trading")
            recs.append("Move 30% of capital to stablecoins (USDT/USDC)")

        if level == "elevated":
            recs.append("Reduce position sizes by 30%")
            recs.append("Tighten stop-losses to 50% of normal")
            recs.append("Increase cash allocation to 40%")

        if level == "moderate":
            recs.append("Monitor correlations closely")
            recs.append("Ensure hedging positions are in place")

        # Specific recommendations based on indicators
        corr_data = indicators.get("correlation", {})
        if corr_data.get("n_high_correlations", 0) > 3:
            recs.append("High correlation detected - diversification benefit reduced")

        vol_data = indicators.get("volatility", {})
        if vol_data.get("regime") == "extreme":
            recs.append("Extreme volatility - use volatility-adjusted position sizing")

        liq_data = indicators.get("liquidity", {})
        if liq_data.get("condition") == "stressed":
            recs.append("Liquidity stress - avoid large orders, use TWAP execution")

        contagion = indicators.get("contagion", {})
        if contagion.get("contagion_ratio", 0) > 0.5:
            recs.append("Contagion risk high - consider exiting all positions")

        return recs

    def get_dashboard_data(self) -> dict:
        """Get data for systemic risk dashboard."""
        assessment = self.assess_systemic_risk()
        return {
            "current_risk": {
                "level": assessment.level,
                "score": assessment.score,
            },
            "indicators": assessment.indicators,
            "recommendations": assessment.recommendations,
            "history": {
                "scores": self._risk_scores[-50:],
                "trend": "worsening" if len(self._risk_scores) > 2 and
                         self._risk_scores[-1] > self._risk_scores[-2] else "improving",
            },
            "timestamp": assessment.timestamp,
        }


# Module-level instance
systemic_risk_monitor = SystemicRiskMonitor()
