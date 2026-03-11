"""Async Binance client wrapper with rate limiting and circuit breaker."""

import asyncio
from decimal import Decimal
from typing import Any

from binance import AsyncClient, BinanceAPIException
from binance.enums import (
    KLINE_INTERVAL_1DAY,
    KLINE_INTERVAL_1HOUR,
    KLINE_INTERVAL_1MINUTE,
    KLINE_INTERVAL_4HOUR,
    KLINE_INTERVAL_5MINUTE,
    KLINE_INTERVAL_15MINUTE,
    ORDER_TYPE_LIMIT,
    ORDER_TYPE_MARKET,
    SIDE_BUY,
    SIDE_SELL,
)
import pandas as pd

from app.config import settings
from app.core.exceptions import (
    ExchangeConnectionError,
    ExchangeRateLimitError,
    InsufficientBalanceError,
    OrderExecutionError,
)
from app.core.logging import get_logger
from app.core.rate_limiter import BinanceRateLimiter
from app.core.resilience import ServiceCircuitBreaker

logger = get_logger(__name__)

INTERVAL_MAP = {
    "1m": KLINE_INTERVAL_1MINUTE,
    "5m": KLINE_INTERVAL_5MINUTE,
    "15m": KLINE_INTERVAL_15MINUTE,
    "1h": KLINE_INTERVAL_1HOUR,
    "4h": KLINE_INTERVAL_4HOUR,
    "1d": KLINE_INTERVAL_1DAY,
}


class BinanceClientWrapper:
    """Async wrapper around python-binance with rate limiting and circuit breaker.

    Uses the shared ``ServiceCircuitBreaker`` from ``app.core.resilience`` for
    circuit-breaking logic, ensuring a single consistent implementation across
    the codebase (CLOSED → OPEN → HALF_OPEN state machine).

    **Rate-limiter note:** The ``BinanceRateLimiter`` tracks request weights and
    order counts using time-based sliding windows.  Ideally we would also
    read the ``X-MBX-USED-WEIGHT-*`` and ``X-MBX-ORDER-COUNT-*`` response
    headers returned by Binance to stay perfectly in sync.  However, the
    ``python-binance`` library processes HTTP responses internally and does
    **not** expose the raw response headers through its async API.  As a
    mitigation we use conservative weight estimates per endpoint and a 20%
    safety margin in the rate-limiter configuration.  If exact header-based
    tracking becomes necessary, a custom ``aiohttp`` session can be injected
    via ``AsyncClient``'s ``session`` parameter.
    """

    def __init__(
        self,
        circuit_breaker: ServiceCircuitBreaker | None = None,
    ) -> None:
        self._client: AsyncClient | None = None
        self._rate_limiter = BinanceRateLimiter()
        self._circuit_breaker = circuit_breaker or ServiceCircuitBreaker(
            failure_threshold=3, recovery_timeout=60.0, half_open_max_calls=2,
        )

    async def connect(self) -> None:
        """Initialize the async Binance client."""
        try:
            if settings.binance_testnet:
                # Manually instantiate to set testnet URL BEFORE ping()
                # AsyncClient.create() pings api.binance.com which is geo-blocked
                self._client = AsyncClient(
                    api_key=settings.active_api_key,
                    api_secret=settings.active_api_secret,
                    testnet=True,
                )
                self._client.API_URL = "https://testnet.binance.vision/api"
                # Manual init that create() would do
                await self._client.ping()
                res = await self._client.get_server_time()
                import time
                self._client.timestamp_offset = res["serverTime"] - int(time.time() * 1000)
            else:
                self._client = await AsyncClient.create(
                    api_key=settings.active_api_key,
                    api_secret=settings.active_api_secret,
                    testnet=False,
                )
            self._circuit_breaker.reset()
            logger.info(
                "binance_connected",
                testnet=settings.binance_testnet,
                api_url=getattr(self._client, "API_URL", "unknown"),
            )
        except Exception as e:
            logger.error("binance_connection_failed", error=str(e))
            raise ExchangeConnectionError(f"Failed to connect to Binance: {e}") from e

    async def disconnect(self) -> None:
        if self._client:
            await self._client.close_connection()
            self._client = None
            logger.info("binance_disconnected")

    async def _execute(self, coro, weight: int = 1) -> Any:
        """Execute a Binance API call with rate limiting and error handling."""
        if not self._circuit_breaker.is_available:
            raise ExchangeConnectionError(
                f"Circuit breaker {self._circuit_breaker.state} - too many consecutive errors"
            )

        if not self._rate_limiter.can_make_request(weight):
            raise ExchangeRateLimitError()

        if not self._client:
            raise ExchangeConnectionError("Client not initialized. Call connect() first.")

        try:
            result = await coro
            self._circuit_breaker.record_success()
            return result
        except BinanceAPIException as e:
            if e.code == -1003:  # Rate limit
                self._circuit_breaker.record_failure()
                raise ExchangeRateLimitError(f"Binance rate limit: {e.message}") from e
            if e.code == -2010:  # Insufficient balance
                raise InsufficientBalanceError(e.message) from e
            self._circuit_breaker.record_failure()
            raise OrderExecutionError(f"Binance API error [{e.code}]: {e.message}") from e
        except Exception as e:
            self._circuit_breaker.record_failure()
            raise ExchangeConnectionError(f"Binance request failed: {e}") from e

    # --- Market Data ---

    async def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 500,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV kline data as a DataFrame."""
        binance_interval = INTERVAL_MAP.get(interval, interval)
        kwargs: dict[str, Any] = {
            "symbol": symbol,
            "interval": binance_interval,
            "limit": limit,
        }
        if start_time:
            kwargs["startTime"] = start_time
        if end_time:
            kwargs["endTime"] = end_time

        raw = await self._execute(self._client.get_klines(**kwargs), weight=2)

        df = pd.DataFrame(
            raw,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trade_count",
                "taker_buy_base", "taker_buy_quote", "ignore",
            ],
        )

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["trade_count"] = df["trade_count"].astype(int)
        df = df[["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trade_count"]]

        return df

    async def get_ticker_price(self, symbol: str) -> Decimal:
        """Get current price for a symbol.

        Tries: 1) authenticated client, 2) testnet API, 3) production mirrors.
        """
        import httpx

        # 1. Try authenticated client first (fastest, already connected)
        if self._client:
            try:
                result = await self._execute(self._client.get_symbol_ticker(symbol=symbol))
                return Decimal(result["price"])
            except Exception:
                pass

        # 2. Try public HTTP endpoints
        mirrors = []
        if settings.binance_testnet:
            mirrors.append(f"https://testnet.binance.vision/api/v3/ticker/price?symbol={symbol}")
        # Production mirrors (may be geo-blocked from US servers)
        mirrors.extend([
            f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}",
            f"https://api1.binance.com/api/v3/ticker/price?symbol={symbol}",
            f"https://api2.binance.com/api/v3/ticker/price?symbol={symbol}",
        ])

        last_error = None
        async with httpx.AsyncClient(timeout=10) as http:
            for url in mirrors:
                try:
                    resp = await http.get(url)
                    if resp.status_code == 200:
                        return Decimal(resp.json()["price"])
                except Exception as e:
                    last_error = e
                    continue

        raise ExchangeConnectionError(f"Cannot get live price for {symbol}: {last_error}")

    async def get_exchange_info(self, symbol: str | None = None) -> dict:
        """Get exchange trading rules and symbol info."""
        if symbol:
            result = await self._execute(self._client.get_symbol_info(symbol), weight=10)
        else:
            result = await self._execute(self._client.get_exchange_info(), weight=20)
        return result

    # --- Account ---

    async def get_account(self) -> dict:
        """Get account balances and info."""
        return await self._execute(self._client.get_account(), weight=20)

    async def get_balance(self, asset: str = "USDT") -> Decimal:
        """Get free balance for a specific asset."""
        account = await self.get_account()
        for balance in account.get("balances", []):
            if balance["asset"] == asset:
                return Decimal(balance["free"])
        return Decimal("0")

    # --- Orders ---

    async def place_market_order(
        self, symbol: str, side: str, quantity: float
    ) -> dict:
        """Place a market order."""
        if not self._rate_limiter.can_place_order():
            raise ExchangeRateLimitError("Order rate limit exceeded")

        params = {
            "symbol": symbol,
            "side": side,
            "type": ORDER_TYPE_MARKET,
            "quantity": f"{quantity:.8f}",
        }
        result = await self._execute(self._client.create_order(**params))
        self._rate_limiter.record_order()

        logger.info(
            "market_order_placed",
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_id=result.get("orderId"),
        )
        return result

    async def place_limit_order(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> dict:
        """Place a limit order."""
        if not self._rate_limiter.can_place_order():
            raise ExchangeRateLimitError("Order rate limit exceeded")

        params = {
            "symbol": symbol,
            "side": side,
            "type": ORDER_TYPE_LIMIT,
            "timeInForce": "GTC",
            "quantity": f"{quantity:.8f}",
            "price": f"{price:.8f}",
        }
        result = await self._execute(self._client.create_order(**params))
        self._rate_limiter.record_order()

        logger.info(
            "limit_order_placed",
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_id=result.get("orderId"),
        )
        return result

    async def cancel_order(self, symbol: str, order_id: int) -> dict:
        """Cancel an open order."""
        result = await self._execute(
            self._client.cancel_order(symbol=symbol, orderId=order_id)
        )
        logger.info("order_cancelled", symbol=symbol, order_id=order_id)
        return result

    async def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        """Get all open orders."""
        kwargs = {}
        if symbol:
            kwargs["symbol"] = symbol
        return await self._execute(self._client.get_open_orders(**kwargs), weight=6)

    async def get_order_status(self, symbol: str, order_id: int) -> dict:
        """Get order status by ID."""
        return await self._execute(
            self._client.get_order(symbol=symbol, orderId=order_id), weight=4
        )


# Global singleton
binance_client = BinanceClientWrapper()
