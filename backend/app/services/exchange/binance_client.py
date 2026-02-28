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
    """Async wrapper around python-binance with rate limiting and circuit breaker."""

    def __init__(self) -> None:
        self._client: AsyncClient | None = None
        self._rate_limiter = BinanceRateLimiter()
        self._consecutive_errors: int = 0
        self._circuit_open: bool = False
        self._circuit_open_until: float = 0

    async def connect(self) -> None:
        """Initialize the async Binance client."""
        try:
            self._client = await AsyncClient.create(
                api_key=settings.active_api_key,
                api_secret=settings.active_api_secret,
                testnet=settings.binance_testnet,
            )
            # For testnet, explicitly set the API URL to bypass geo-restrictions
            if settings.binance_testnet:
                self._client.API_URL = "https://testnet.binance.vision/api"
            # Verify connection
            await self._client.get_server_time()
            self._consecutive_errors = 0
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

    def _check_circuit_breaker(self) -> None:
        """Check if circuit breaker is open."""
        if self._circuit_open:
            import time

            if time.monotonic() < self._circuit_open_until:
                raise ExchangeConnectionError(
                    "Circuit breaker open - too many consecutive errors"
                )
            self._circuit_open = False
            self._consecutive_errors = 0
            logger.info("circuit_breaker_reset")

    def _record_error(self) -> None:
        import time

        self._consecutive_errors += 1
        if self._consecutive_errors >= 3:
            self._circuit_open = True
            self._circuit_open_until = time.monotonic() + 60  # 60s cooldown
            logger.warning(
                "circuit_breaker_triggered",
                consecutive_errors=self._consecutive_errors,
            )

    def _record_success(self) -> None:
        self._consecutive_errors = 0

    async def _execute(self, coro, weight: int = 1) -> Any:
        """Execute a Binance API call with rate limiting and error handling."""
        self._check_circuit_breaker()

        if not self._rate_limiter.can_make_request(weight):
            raise ExchangeRateLimitError()

        if not self._client:
            raise ExchangeConnectionError("Client not initialized. Call connect() first.")

        try:
            result = await coro
            self._record_success()
            return result
        except BinanceAPIException as e:
            if e.code == -1003:  # Rate limit
                self._record_error()
                raise ExchangeRateLimitError(f"Binance rate limit: {e.message}") from e
            if e.code == -2010:  # Insufficient balance
                raise InsufficientBalanceError(e.message) from e
            self._record_error()
            raise OrderExecutionError(f"Binance API error [{e.code}]: {e.message}") from e
        except Exception as e:
            self._record_error()
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
        """Get current price for a symbol."""
        result = await self._execute(self._client.get_symbol_ticker(symbol=symbol))
        return Decimal(result["price"])

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
