"""Base exchange adapter interface.

Defines the unified contract that all exchange integrations must implement.
This enables multi-exchange support with a single trading engine.
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any

import pandas as pd


class ExchangeAdapter(ABC):
    """Unified interface for all exchange integrations.

    Every adapter (Binance, Bybit, OKX, etc.) must implement these methods.
    The trading engine, order manager, and risk system depend on this contract.

    Methods are grouped by concern:
    - Connection lifecycle: connect, disconnect, is_connected
    - Market data: get_ticker_price, get_klines
    - Account: get_balance
    - Orders: place_order, place_market_order, cancel_order, get_open_orders
    - Identity: name, get_exchange_name
    """

    # --- Connection lifecycle ---

    @abstractmethod
    async def connect(self) -> None:
        """Initialize the exchange connection (API keys, WebSocket, etc.)."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully close the exchange connection."""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if the adapter has an active connection."""
        ...

    # --- Market data ---

    @abstractmethod
    async def get_ticker_price(self, symbol: str) -> Decimal:
        """Get current price for a trading pair.

        Args:
            symbol: Trading pair (e.g. "BTCUSDT").

        Returns:
            Current price as Decimal for financial precision.

        Raises:
            ExchangeConnectionError: If price cannot be retrieved.
        """
        ...

    @abstractmethod
    async def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch OHLCV candlestick data.

        Args:
            symbol: Trading pair (e.g. "BTCUSDT").
            interval: Candle interval ("1m", "5m", "15m", "1h", "4h", "1d").
            limit: Maximum number of candles to return.

        Returns:
            DataFrame with columns: open_time, open, high, low, close, volume.
        """
        ...

    # --- Account ---

    @abstractmethod
    async def get_balance(self, asset: str = "USDT") -> Decimal:
        """Get available (free) balance for an asset.

        Args:
            asset: Asset symbol (e.g. "USDT", "BTC").

        Returns:
            Free balance as Decimal.
        """
        ...

    # --- Orders ---

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: float | None = None,
    ) -> dict[str, Any]:
        """Place an order on the exchange.

        Unified order method that handles both MARKET and LIMIT orders.

        Args:
            symbol: Trading pair (e.g. "BTCUSDT").
            side: "BUY" or "SELL".
            quantity: Order quantity in base asset.
            order_type: "MARKET" or "LIMIT".
            price: Required for LIMIT orders, ignored for MARKET.

        Returns:
            Dict with at least: orderId, status, executedQty, avgPrice.

        Raises:
            OrderExecutionError: If order placement fails.
            InsufficientBalanceError: If account has insufficient funds.
        """
        ...

    @abstractmethod
    async def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
    ) -> dict[str, Any]:
        """Convenience method: place a market order.

        Equivalent to place_order(symbol, side, quantity, order_type="MARKET").
        """
        ...

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str | int) -> dict[str, Any]:
        """Cancel an open order.

        Args:
            symbol: Trading pair.
            order_id: Exchange-specific order identifier.

        Returns:
            Dict with cancellation confirmation.
        """
        ...

    @abstractmethod
    async def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Get all open orders, optionally filtered by symbol.

        Args:
            symbol: If provided, only return orders for this pair.

        Returns:
            List of order dicts.
        """
        ...

    # --- Identity ---

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name of the exchange (e.g. 'binance', 'bybit')."""
        ...

    def get_exchange_name(self) -> str:
        """Return the exchange name. Delegates to the `name` property."""
        return self.name

    # --- Paper trading support ---

    @property
    def supports_paper_trading(self) -> bool:
        """Whether this adapter supports simulated paper trading.

        Override in subclasses that implement paper mode.
        """
        return False


# Keep backward compatibility alias
IExchangeAdapter = ExchangeAdapter
