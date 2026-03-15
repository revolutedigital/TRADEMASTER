"""Systemic risk monitoring: cross-asset correlation and market-wide risk signals."""
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SystemicRiskReport:
    risk_level: str  # "low", "elevated", "high", "critical"
    risk_score: float  # 0-100
    correlation_avg: float
    volatility_regime: str  # "low", "normal", "high", "extreme"
    signals: list[dict]
    timestamp: str


class SystemicRiskMonitor:
    """Monitor systemic risk across the crypto market."""

    def analyze(
        self,
        price_series: dict[str, np.ndarray],  # {symbol: close_prices}
        lookback: int = 30,
    ) -> SystemicRiskReport:
        """Analyze systemic risk from multiple asset price series."""
        signals = []

        # 1. Cross-asset correlation
        returns = {}
        for symbol, prices in price_series.items():
            if len(prices) >= lookback + 1:
                r = np.diff(prices[-lookback-1:]) / prices[-lookback-1:-1]
                returns[symbol] = r

        correlations = []
        symbols = list(returns.keys())
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                if len(returns[symbols[i]]) == len(returns[symbols[j]]):
                    corr = np.corrcoef(returns[symbols[i]], returns[symbols[j]])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)

        avg_corr = float(np.mean(correlations)) if correlations else 0.0

        if avg_corr > 0.85:
            signals.append({
                "type": "high_correlation",
                "severity": "high",
                "message": f"Cross-asset correlation at {avg_corr:.2f} - assets moving together (contagion risk)",
            })
        elif avg_corr > 0.7:
            signals.append({
                "type": "elevated_correlation",
                "severity": "medium",
                "message": f"Correlation at {avg_corr:.2f} - moderate herding behavior",
            })

        # 2. Market-wide volatility
        all_vols = []
        for symbol, r in returns.items():
            vol = float(np.std(r)) * np.sqrt(252) if len(r) > 1 else 0
            all_vols.append(vol)

        avg_vol = float(np.mean(all_vols)) if all_vols else 0

        if avg_vol > 1.5:
            vol_regime = "extreme"
            signals.append({
                "type": "extreme_volatility",
                "severity": "critical",
                "message": f"Annualized volatility at {avg_vol:.0%} - extreme market conditions",
            })
        elif avg_vol > 0.8:
            vol_regime = "high"
            signals.append({
                "type": "high_volatility",
                "severity": "high",
                "message": f"Annualized volatility at {avg_vol:.0%} - elevated risk",
            })
        elif avg_vol > 0.4:
            vol_regime = "normal"
        else:
            vol_regime = "low"

        # 3. Drawdown across assets
        for symbol, prices in price_series.items():
            if len(prices) >= 10:
                peak = np.max(prices[-30:]) if len(prices) >= 30 else np.max(prices)
                current = prices[-1]
                dd = (peak - current) / peak if peak > 0 else 0
                if dd > 0.20:
                    signals.append({
                        "type": "deep_drawdown",
                        "severity": "high",
                        "message": f"{symbol} in {dd:.0%} drawdown from recent peak",
                    })

        # 4. Calculate overall risk score (0-100)
        risk_score = 0.0
        risk_score += min(30, avg_corr * 35)  # Correlation contribution (max 30)
        risk_score += min(40, avg_vol * 30)    # Volatility contribution (max 40)
        risk_score += min(30, len([s for s in signals if s["severity"] in ("high", "critical")]) * 10)
        risk_score = min(100, risk_score)

        if risk_score >= 75:
            risk_level = "critical"
        elif risk_score >= 50:
            risk_level = "high"
        elif risk_score >= 25:
            risk_level = "elevated"
        else:
            risk_level = "low"

        return SystemicRiskReport(
            risk_level=risk_level,
            risk_score=round(risk_score, 1),
            correlation_avg=round(avg_corr, 4),
            volatility_regime=vol_regime,
            signals=signals,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


systemic_risk_monitor = SystemicRiskMonitor()
