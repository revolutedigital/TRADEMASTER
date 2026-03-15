"""Market sentiment analysis from multiple sources."""
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SentimentScore:
    overall: float  # -1.0 (extreme fear) to 1.0 (extreme greed)
    confidence: float  # 0-1
    sources: dict[str, float]  # Per-source scores
    timestamp: str
    interpretation: str  # "extreme_fear", "fear", "neutral", "greed", "extreme_greed"


class SentimentAnalyzer:
    """Analyze market sentiment from available data sources.

    Uses on-chain and market microstructure signals rather than
    external NLP APIs, so it works fully offline.
    """

    # Funding rate thresholds (perpetual futures)
    FUNDING_BULLISH = 0.01  # > 1% = excessive greed
    FUNDING_BEARISH = -0.005  # < -0.5% = fear

    def analyze_from_market_data(
        self,
        prices: np.ndarray,  # Recent close prices
        volumes: np.ndarray,  # Recent volumes
        funding_rate: float | None = None,
        open_interest_change: float | None = None,
    ) -> SentimentScore:
        """Derive sentiment from market microstructure."""
        sources: dict[str, float] = {}

        if len(prices) >= 20:
            # 1. Price momentum: SMA ratio
            sma_fast = np.mean(prices[-5:])
            sma_slow = np.mean(prices[-20:])
            momentum = (sma_fast / sma_slow - 1) * 10  # Scale to roughly -1..1
            sources["momentum"] = float(np.clip(momentum, -1, 1))

            # 2. Volatility regime: high vol = fear
            returns = np.diff(prices) / prices[:-1]
            recent_vol = np.std(returns[-10:]) if len(returns) >= 10 else 0
            hist_vol = np.std(returns) if len(returns) >= 2 else recent_vol
            vol_ratio = recent_vol / hist_vol if hist_vol > 0 else 1.0
            # High vol = fear (-1), low vol = complacency (slight positive)
            vol_sentiment = float(np.clip(1 - vol_ratio, -1, 1))
            sources["volatility"] = vol_sentiment

            # 3. Volume trend: increasing volume on up days = bullish
            if len(volumes) >= 10:
                recent_avg_vol = np.mean(volumes[-5:])
                older_avg_vol = np.mean(volumes[-10:-5])
                vol_change = (recent_avg_vol / older_avg_vol - 1) if older_avg_vol > 0 else 0
                price_direction = 1 if prices[-1] > prices[-5] else -1
                vol_signal = float(np.clip(vol_change * price_direction, -1, 1))
                sources["volume_trend"] = vol_signal

        # 4. Funding rate (if available)
        if funding_rate is not None:
            if funding_rate > self.FUNDING_BULLISH:
                sources["funding"] = -0.5  # Too greedy = contrarian bearish
            elif funding_rate < self.FUNDING_BEARISH:
                sources["funding"] = 0.5  # Too fearful = contrarian bullish
            else:
                sources["funding"] = float(np.clip(funding_rate * 50, -1, 1))

        # 5. Open interest change
        if open_interest_change is not None:
            sources["open_interest"] = float(np.clip(open_interest_change * 5, -1, 1))

        # Weighted average
        weights = {
            "momentum": 0.35,
            "volatility": 0.25,
            "volume_trend": 0.20,
            "funding": 0.10,
            "open_interest": 0.10,
        }

        total_weight = sum(weights.get(k, 0.1) for k in sources)
        if total_weight > 0:
            overall = sum(sources[k] * weights.get(k, 0.1) for k in sources) / total_weight
        else:
            overall = 0.0

        overall = float(np.clip(overall, -1, 1))
        confidence = min(1.0, len(sources) / 4)  # More sources = more confident

        # Interpretation
        if overall <= -0.6:
            interpretation = "extreme_fear"
        elif overall <= -0.2:
            interpretation = "fear"
        elif overall <= 0.2:
            interpretation = "neutral"
        elif overall <= 0.6:
            interpretation = "greed"
        else:
            interpretation = "extreme_greed"

        return SentimentScore(
            overall=round(overall, 4),
            confidence=round(confidence, 4),
            sources={k: round(v, 4) for k, v in sources.items()},
            timestamp=datetime.now(timezone.utc).isoformat(),
            interpretation=interpretation,
        )


sentiment_analyzer = SentimentAnalyzer()
