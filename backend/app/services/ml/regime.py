"""Market regime detection using statistical methods."""
import numpy as np
from dataclasses import dataclass
from enum import Enum

from app.core.logging import get_logger

logger = get_logger(__name__)


class MarketRegime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"


@dataclass
class RegimeResult:
    regime: MarketRegime
    confidence: float
    volatility: float
    trend_strength: float
    duration_candles: int


class RegimeDetector:
    """Detect market regime using returns, volatility, and trend analysis.

    Uses statistical methods (no HMM dependency required):
    - Returns distribution for bull/bear classification
    - Volatility percentile for volatility regime
    - ADX-like trend strength for sideways detection
    """

    def detect(self, returns: np.ndarray, lookback: int = 50) -> RegimeResult:
        """Detect current market regime from return series."""
        if len(returns) < lookback:
            return RegimeResult(
                regime=MarketRegime.SIDEWAYS,
                confidence=0.3,
                volatility=0.0,
                trend_strength=0.0,
                duration_candles=0,
            )

        recent = returns[-lookback:]
        mean_return = np.mean(recent)
        volatility = np.std(recent)
        cumulative = np.sum(recent)

        # Historical volatility percentile
        full_vol = np.std(returns)
        vol_ratio = volatility / full_vol if full_vol > 0 else 1.0

        # Trend strength: ratio of directional move vs volatility
        trend_strength = abs(cumulative) / (volatility * np.sqrt(lookback)) if volatility > 0 else 0

        # Classify regime
        if vol_ratio > 1.5:
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = min(0.9, vol_ratio / 2)
        elif trend_strength > 1.5 and mean_return > 0:
            regime = MarketRegime.BULL
            confidence = min(0.9, trend_strength / 3)
        elif trend_strength > 1.5 and mean_return < 0:
            regime = MarketRegime.BEAR
            confidence = min(0.9, trend_strength / 3)
        else:
            regime = MarketRegime.SIDEWAYS
            confidence = 1.0 - min(0.7, trend_strength / 2)

        # Estimate regime duration
        duration = self._estimate_duration(returns, regime, lookback)

        return RegimeResult(
            regime=regime,
            confidence=round(confidence, 3),
            volatility=round(float(volatility), 6),
            trend_strength=round(float(trend_strength), 3),
            duration_candles=duration,
        )

    def _estimate_duration(self, returns: np.ndarray, current_regime: MarketRegime, lookback: int) -> int:
        """Estimate how long the current regime has been active."""
        duration = 0
        for i in range(len(returns) - 1, max(0, len(returns) - lookback * 3), -1):
            window = returns[max(0, i - lookback):i]
            if len(window) < 10:
                break
            mean_r = np.mean(window)
            if current_regime == MarketRegime.BULL and mean_r > 0:
                duration += 1
            elif current_regime == MarketRegime.BEAR and mean_r < 0:
                duration += 1
            elif current_regime == MarketRegime.SIDEWAYS and abs(mean_r) < np.std(window) * 0.5:
                duration += 1
            else:
                break
        return duration


regime_detector = RegimeDetector()
