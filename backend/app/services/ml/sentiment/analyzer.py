"""Sentiment analysis using FinBERT for crypto news."""

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SentimentResult:
    """Result of sentiment analysis on text."""

    text: str
    positive: float
    negative: float
    neutral: float
    compound: float  # [-1.0, +1.0] overall sentiment
    label: str  # "positive", "negative", "neutral"


class SentimentAnalyzer:
    """FinBERT-based sentiment analyzer for financial news.

    Uses the ProsusAI/finbert model to analyze sentiment of
    crypto-related news headlines and articles.
    """

    def __init__(self):
        self._pipeline = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self) -> None:
        """Load the FinBERT model. Deferred to avoid slow startup."""
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                top_k=None,
                truncation=True,
                max_length=512,
            )
            self._is_loaded = True
            logger.info("finbert_loaded")
        except Exception as e:
            logger.error("finbert_load_failed", error=str(e))
            self._is_loaded = False

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of a single text."""
        if not self._is_loaded or self._pipeline is None:
            # Fallback: neutral
            return SentimentResult(
                text=text,
                positive=0.33,
                negative=0.33,
                neutral=0.34,
                compound=0.0,
                label="neutral",
            )

        results = self._pipeline(text)[0]
        scores = {r["label"]: r["score"] for r in results}

        positive = scores.get("positive", 0)
        negative = scores.get("negative", 0)
        neutral = scores.get("neutral", 0)
        compound = positive - negative

        label = max(scores, key=scores.get)  # type: ignore[arg-type]

        return SentimentResult(
            text=text[:200],
            positive=positive,
            negative=negative,
            neutral=neutral,
            compound=compound,
            label=label,
        )

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Analyze sentiment of multiple texts."""
        return [self.analyze(t) for t in texts]

    def aggregate_sentiment(self, results: list[SentimentResult]) -> float:
        """Aggregate multiple sentiment results into a single score.

        Returns: float in [-1.0, +1.0]
            Weighted by recency (more recent = higher weight).
        """
        if not results:
            return 0.0

        # Exponential decay weighting (newest first)
        n = len(results)
        weights = np.exp(-np.arange(n) * 0.3)
        weights /= weights.sum()

        compounds = np.array([r.compound for r in results])
        return float(np.dot(compounds, weights))


# Module-level singleton
sentiment_analyzer = SentimentAnalyzer()
