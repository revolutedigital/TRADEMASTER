"""Bybit exchange adapter (stub for multi-exchange support)."""
from decimal import Decimal
import pandas as pd

from app.services.exchange.adapters.base import IExchangeAdapter
from app.core.logging import get_logger

logger = get_logger(__name__)


class BybitAdapter(IExchangeAdapter):
    """Bybit exchange adapter - stub implementation for future integration."""

    def __init__(self):
        self._connected = False

    async def connect(self) -> None:
        logger.info("bybit_adapter_connect_stub")
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def get_ticker_price(self, symbol: str) -> Decimal:
        raise NotImplementedError("Bybit adapter not yet implemented")

    async def get_klines(self, symbol: str, interval: str = "1h", limit: int = 500) -> pd.DataFrame:
        raise NotImplementedError("Bybit adapter not yet implemented")

    async def get_balance(self, asset: str = "USDT") -> Decimal:
        raise NotImplementedError("Bybit adapter not yet implemented")

    async def place_market_order(self, symbol: str, side: str, quantity: float) -> dict:
        raise NotImplementedError("Bybit adapter not yet implemented")

    async def cancel_order(self, symbol: str, order_id: int) -> dict:
        raise NotImplementedError("Bybit adapter not yet implemented")

    async def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        raise NotImplementedError("Bybit adapter not yet implemented")

    @property
    def name(self) -> str:
        return "bybit"

    @property
    def is_connected(self) -> bool:
        return self._connected
