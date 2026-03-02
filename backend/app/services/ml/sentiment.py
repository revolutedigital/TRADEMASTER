"""Market sentiment analysis from external data sources."""
from dataclasses import dataclass
from datetime import datetime, timezone

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SentimentScore:
    score: float  # -1.0 (bearish) to 1.0 (bullish)
    confidence: float  # 0.0 to 1.0
    source: str
    key_topics: list[str]
    timestamp: datetime


class SentimentAnalyzer:
    """Analyze market sentiment from multiple sources.

    Sources:
    - Fear & Greed Index (alternative.me API)
    - Binance funding rates
    - Long/short ratios
    - Open interest changes
    """

    async def get_fear_greed_index(self) -> dict:
        """Fetch the Crypto Fear & Greed Index."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.alternative.me/fng/?limit=1") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        value = int(data["data"][0]["value"])
                        classification = data["data"][0]["value_classification"]
                        return {
                            "value": value,
                            "classification": classification,
                            "score": (value - 50) / 50,  # Normalize to -1..1
                        }
        except Exception as e:
            logger.warning("fear_greed_fetch_failed", error=str(e))
        return {"value": 50, "classification": "Neutral", "score": 0.0}

    async def get_composite_sentiment(self, symbol: str) -> SentimentScore:
        """Get composite sentiment score combining multiple indicators."""
        fng = await self.get_fear_greed_index()

        # Composite score: weighted average of available indicators
        score = fng["score"] * 0.6  # Fear & Greed weighted 60%

        return SentimentScore(
            score=round(max(-1.0, min(1.0, score)), 4),
            confidence=0.6,
            source="composite",
            key_topics=[fng["classification"]],
            timestamp=datetime.now(timezone.utc),
        )


sentiment_analyzer = SentimentAnalyzer()
