"""Binance exchange adapter implementing ExchangeAdapter.

Wraps the existing BinanceClientWrapper singleton to conform to the
multi-exchange adapter interface. All calls delegate to the real client,
preserving rate limiting, circuit breaker, and geo-block handling.
"""

from decimal import Decimal
from typing import Any

import pandas as pd

from app.core.logging import get_logger
from app.services.exchange.adapters.base import ExchangeAdapter
from app.services.exchange.binance_client import binance_client, BinanceClientWrapper

logger = get_logger(__name__)


class BinanceAdapter(ExchangeAdapter):
    """Adapter wrapping the existing BinanceClientWrapper singleton.

    This adapter:
    - Delegates all calls to the battle-tested BinanceClientWrapper
    - Preserves rate limiting, circuit breaker, Redis price cache
    - Supports paper trading via the existing order_manager flow
    - Handles geo-blocked servers (Railway US) gracefully
    """

    def __init__(self, client: BinanceClientWrapper | None = None) -> None:
        """Initialize with optional custom client (useful for testing).

        Args:
            client: BinanceClientWrapper instance. Defaults to global singleton.
        """
        self._client = client or binance_client

    # --- Connection lifecycle ---

    async def connect(self) -> None:
        """Connect to Binance API.

        May fail on geo-blocked servers -- this is expected.
        Paper trading still works via frontend-supplied prices.
        """
        try:
            await self._client.connect()
        except Exception as e:
            logger.warning("binance_adapter_connect_failed", error=str(e))
            raise

    async def disconnect(self) -> None:
        await self._client.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._client._client is not None

    # --- Market data ---

    async def get_ticker_price(self, symbol: str) -> Decimal:
        """Get current price. Uses Redis cache -> client -> HTTP mirrors."""
        return await self._client.get_ticker_price(symbol)

    async def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch OHLCV candle data as DataFrame."""
        return await self._client.get_klines(symbol, interval, limit)

    # --- Account ---

    async def get_balance(self, asset: str = "USDT") -> Decimal:
        """Get free balance for an asset."""
        return await self._client.get_balance(asset)

    # --- Orders ---

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: float | None = None,
    ) -> dict[str, Any]:
        """Place an order on Binance.

        Routes to market or limit order based on order_type.
        """
        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("Price is required for LIMIT orders")
            return await self._client.place_limit_order(symbol, side, quantity, price)
        return await self._client.place_market_order(symbol, side, quantity)

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
    ) -> dict[str, Any]:
        """Place a market order on Binance."""
        return await self._client.place_market_order(symbol, side, quantity)

    async def cancel_order(self, symbol: str, order_id: str | int) -> dict[str, Any]:
        """Cancel an open order on Binance."""
        return await self._client.cancel_order(symbol, int(order_id))

    async def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Get all open orders on Binance."""
        return await self._client.get_open_orders(symbol)

    # --- Identity ---

    @property
    def name(self) -> str:
        return "binance"

    @property
    def supports_paper_trading(self) -> bool:
        """Binance adapter supports paper trading via order_manager."""
        return True
