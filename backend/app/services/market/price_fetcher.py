"""Autonomous price fetcher — backend fetches prices without needing the frontend open.

Uses free public APIs (CoinGecko, CryptoCompare) that are NOT geo-blocked,
stores prices in Redis so the synthetic kline generator can build candles.
This replaces the frontend-relay dependency for price data.
"""

import asyncio
import httpx

from app.config import settings
from app.core.events import event_bus
from app.core.logging import get_logger

logger = get_logger(__name__)

# Map TradeMaster symbols to CoinGecko IDs
_COINGECKO_IDS = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "BNBUSDT": "binancecoin",
    "SOLUSDT": "solana",
    "XRPUSDT": "ripple",
    "DOGEUSDT": "dogecoin",
    "ADAUSDT": "cardano",
    "AVAXUSDT": "avalanche-2",
    "DOTUSDT": "polkadot",
    "LINKUSDT": "chainlink",
}

# Map TradeMaster symbols to CryptoCompare fsyms
_CC_SYMBOLS = {
    "BTCUSDT": "BTC",
    "ETHUSDT": "ETH",
    "BNBUSDT": "BNB",
    "SOLUSDT": "SOL",
    "XRPUSDT": "XRP",
    "DOGEUSDT": "DOGE",
    "ADAUSDT": "ADA",
    "AVAXUSDT": "AVAX",
    "DOTUSDT": "DOT",
    "LINKUSDT": "LINK",
}


class PriceFetcher:
    """Fetches prices from public APIs and stores them in Redis."""

    def __init__(self) -> None:
        self._running: bool = False
        self._interval: int = 10  # seconds between fetches
        self._client: httpx.AsyncClient | None = None
        self._consecutive_failures: int = 0
        self._source: str = "unknown"

    async def start(self) -> None:
        self._running = True
        self._client = httpx.AsyncClient(timeout=15, follow_redirects=True)
        logger.info(
            "price_fetcher_starting",
            symbols=settings.symbols_list,
            interval=self._interval,
        )

        while self._running:
            try:
                fetched = await self._fetch_and_store()
                if fetched:
                    self._consecutive_failures = 0
                else:
                    self._consecutive_failures += 1
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._consecutive_failures += 1
                logger.error("price_fetcher_error", error=str(e))

            # Back off on repeated failures (max 60s)
            wait = min(self._interval * (1 + self._consecutive_failures), 60)
            await asyncio.sleep(wait)

        if self._client:
            await self._client.aclose()
        logger.info("price_fetcher_stopped")

    async def stop(self) -> None:
        self._running = False

    async def _fetch_and_store(self) -> bool:
        """Try multiple sources. Returns True if at least one price was stored."""
        symbols = settings.symbols_list

        # Try CoinGecko first (no API key needed, 30 req/min free)
        prices = await self._fetch_coingecko(symbols)
        if prices:
            self._source = "coingecko"
            await self._store_prices(prices)
            return True

        # Fallback: CryptoCompare
        prices = await self._fetch_cryptocompare(symbols)
        if prices:
            self._source = "cryptocompare"
            await self._store_prices(prices)
            return True

        return False

    async def _fetch_coingecko(self, symbols: list[str]) -> dict[str, float]:
        """Fetch prices from CoinGecko free API."""
        ids = [_COINGECKO_IDS[s] for s in symbols if s in _COINGECKO_IDS]
        if not ids:
            return {}

        try:
            resp = await self._client.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": ",".join(ids), "vs_currencies": "usd"},
            )
            if resp.status_code == 429:
                logger.warning("coingecko_rate_limited")
                return {}
            if resp.status_code != 200:
                return {}

            data = resp.json()
            result = {}
            for symbol in symbols:
                cg_id = _COINGECKO_IDS.get(symbol)
                if cg_id and cg_id in data and "usd" in data[cg_id]:
                    result[symbol] = float(data[cg_id]["usd"])
            return result
        except Exception as e:
            logger.debug("coingecko_fetch_failed", error=str(e))
            return {}

    async def _fetch_cryptocompare(self, symbols: list[str]) -> dict[str, float]:
        """Fetch prices from CryptoCompare free API."""
        fsyms = [_CC_SYMBOLS[s] for s in symbols if s in _CC_SYMBOLS]
        if not fsyms:
            return {}

        try:
            resp = await self._client.get(
                "https://min-api.cryptocompare.com/data/pricemulti",
                params={"fsyms": ",".join(fsyms), "tsyms": "USD"},
            )
            if resp.status_code != 200:
                return {}

            data = resp.json()
            result = {}
            for symbol in symbols:
                cc_sym = _CC_SYMBOLS.get(symbol)
                if cc_sym and cc_sym in data and "USD" in data[cc_sym]:
                    result[symbol] = float(data[cc_sym]["USD"])
            return result
        except Exception as e:
            logger.debug("cryptocompare_fetch_failed", error=str(e))
            return {}

    async def _store_prices(self, prices: dict[str, float]) -> None:
        """Store prices in Redis with 30s TTL."""
        if not event_bus._redis:
            return

        for symbol, price in prices.items():
            if price > 0:
                await event_bus._redis.set(f"price:{symbol}", str(price), ex=60)

        logger.debug(
            "prices_stored",
            source=self._source,
            count=len(prices),
            symbols=list(prices.keys()),
        )


price_fetcher = PriceFetcher()
