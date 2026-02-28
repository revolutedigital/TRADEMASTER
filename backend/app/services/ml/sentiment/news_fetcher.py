"""News fetcher for crypto sentiment analysis."""

from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from app.core.logging import get_logger

logger = get_logger(__name__)

CRYPTO_NEWS_SOURCES = [
    "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&categories=BTC,ETH",
]


@dataclass
class NewsItem:
    """A news article for sentiment analysis."""

    title: str
    source: str
    published_at: datetime
    url: str
    body: str = ""


class NewsFetcher:
    """Fetches crypto news headlines for sentiment analysis."""

    def __init__(self, timeout: float = 10.0):
        self._timeout = timeout

    async def fetch_latest(self, limit: int = 20) -> list[NewsItem]:
        """Fetch latest crypto news from CryptoCompare API."""
        items: list[NewsItem] = []

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(CRYPTO_NEWS_SOURCES[0])
                resp.raise_for_status()
                data = resp.json()

                for article in data.get("Data", [])[:limit]:
                    items.append(
                        NewsItem(
                            title=article.get("title", ""),
                            source=article.get("source", ""),
                            published_at=datetime.fromtimestamp(
                                article.get("published_on", 0),
                                tz=timezone.utc,
                            ),
                            url=article.get("url", ""),
                            body=article.get("body", "")[:500],
                        )
                    )

            logger.info("news_fetched", count=len(items))
        except Exception as e:
            logger.error("news_fetch_failed", error=str(e))

        return items


news_fetcher = NewsFetcher()
