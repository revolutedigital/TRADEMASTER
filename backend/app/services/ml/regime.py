"""Market regime detection: bull / bear / sideways + volatility classification.

Uses price action + volatility to classify the regime.
Provides adaptive parameters (signal threshold, position sizing, exposure limits)
that change based on detected regime.

Hysteresis prevents rapid regime switching (15min minimum between changes).
"""

import time
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


class MarketRegime(StrEnum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"


class VolatilityRegime(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


@dataclass
class RegimeState:
    """Current detected regime with adaptive parameters."""

    market: MarketRegime
    volatility: VolatilityRegime
    trend_slope: float  # Normalized slope
    vol_ratio: float  # Current vol / historical vol
    confidence: float

    # Adaptive parameters
    signal_threshold: float
    position_size_mult: float
    max_exposure_mult: float


# Regime-specific parameter presets
_REGIME_PARAMS = {
    (MarketRegime.BULL, VolatilityRegime.LOW): dict(signal_threshold=0.20, position_size_mult=1.2, max_exposure_mult=1.0),
    (MarketRegime.BULL, VolatilityRegime.NORMAL): dict(signal_threshold=0.25, position_size_mult=1.0, max_exposure_mult=1.0),
    (MarketRegime.BULL, VolatilityRegime.HIGH): dict(signal_threshold=0.30, position_size_mult=0.8, max_exposure_mult=0.8),
    (MarketRegime.SIDEWAYS, VolatilityRegime.LOW): dict(signal_threshold=0.35, position_size_mult=0.6, max_exposure_mult=0.7),
    (MarketRegime.SIDEWAYS, VolatilityRegime.NORMAL): dict(signal_threshold=0.30, position_size_mult=0.7, max_exposure_mult=0.8),
    (MarketRegime.SIDEWAYS, VolatilityRegime.HIGH): dict(signal_threshold=0.35, position_size_mult=0.5, max_exposure_mult=0.6),
    (MarketRegime.BEAR, VolatilityRegime.LOW): dict(signal_threshold=0.30, position_size_mult=0.7, max_exposure_mult=0.7),
    (MarketRegime.BEAR, VolatilityRegime.NORMAL): dict(signal_threshold=0.35, position_size_mult=0.5, max_exposure_mult=0.6),
    (MarketRegime.BEAR, VolatilityRegime.HIGH): dict(signal_threshold=0.40, position_size_mult=0.3, max_exposure_mult=0.4),
}


class RegimeDetector:
    """Detects market regime from price data and provides adaptive parameters."""

    def __init__(self, lookback: int = 50, hysteresis_seconds: int = 900) -> None:
        self._lookback = lookback
        self._hysteresis = hysteresis_seconds
        self._current: dict[str, RegimeState] = {}
        self._last_change: dict[str, float] = {}

    def detect(self, close_prices: np.ndarray, symbol: str = "default") -> RegimeState:
        """Detect current regime from close prices array."""
        n = len(close_prices)
        lookback = min(self._lookback, n - 1)

        if lookback < 10:
            return self._default_state()

        prices = close_prices[-lookback:]

        # 1. Trend: linear regression slope normalized by price
        x = np.arange(lookback, dtype=float)
        slope = float(np.polyfit(x, prices, 1)[0])
        avg_price = float(np.mean(prices))
        norm_slope = (slope * lookback) / avg_price if avg_price > 0 else 0

        # 2. Volatility regime
        if n >= lookback + 1:
            returns = np.diff(close_prices[-lookback - 1:]) / close_prices[-lookback - 1:-1]
            recent_vol = float(np.std(returns[-14:])) if len(returns) >= 14 else float(np.std(returns))
            full_vol = float(np.std(returns))
            vol_ratio = recent_vol / full_vol if full_vol > 0 else 1.0
        else:
            vol_ratio = 1.0

        vol_regime = VolatilityRegime.LOW if vol_ratio < 0.7 else (
            VolatilityRegime.HIGH if vol_ratio > 1.3 else VolatilityRegime.NORMAL
        )

        # 3. Classify market regime
        if norm_slope > 0.02:
            market = MarketRegime.BULL
            confidence = min(abs(norm_slope) / 0.10, 1.0)
        elif norm_slope < -0.02:
            market = MarketRegime.BEAR
            confidence = min(abs(norm_slope) / 0.10, 1.0)
        else:
            market = MarketRegime.SIDEWAYS
            confidence = 1.0 - min(abs(norm_slope) / 0.02, 1.0)

        # Hysteresis: prevent rapid switching
        now = time.time()
        prev = self._current.get(symbol)
        if prev and prev.market != market:
            if now - self._last_change.get(symbol, 0) < self._hysteresis:
                market = prev.market

        params = _REGIME_PARAMS.get(
            (market, vol_regime),
            dict(signal_threshold=0.25, position_size_mult=1.0, max_exposure_mult=1.0),
        )

        state = RegimeState(
            market=market,
            volatility=vol_regime,
            trend_slope=round(float(norm_slope), 6),
            vol_ratio=round(vol_ratio, 4),
            confidence=round(confidence, 4),
            **params,
        )

        if prev is None or prev.market != state.market:
            self._last_change[symbol] = now
            if prev:
                logger.info(
                    "regime_change",
                    symbol=symbol,
                    prev=prev.market,
                    new=state.market,
                    volatility=state.volatility,
                    slope=state.trend_slope,
                )

        self._current[symbol] = state
        return state

    def get_current(self, symbol: str = "default") -> RegimeState:
        return self._current.get(symbol, self._default_state())

    def get_all(self) -> dict[str, dict]:
        """Get all regimes for API display."""
        return {
            sym: {
                "market": s.market,
                "volatility": s.volatility,
                "trend_slope": s.trend_slope,
                "confidence": s.confidence,
                "signal_threshold": s.signal_threshold,
                "position_size_mult": s.position_size_mult,
                "max_exposure_mult": s.max_exposure_mult,
            }
            for sym, s in self._current.items()
        }

    @staticmethod
    def _default_state() -> RegimeState:
        return RegimeState(
            market=MarketRegime.SIDEWAYS,
            volatility=VolatilityRegime.NORMAL,
            trend_slope=0,
            vol_ratio=1.0,
            confidence=0,
            signal_threshold=0.25,
            position_size_mult=1.0,
            max_exposure_mult=1.0,
        )


regime_detector = RegimeDetector()
