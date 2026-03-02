"""Binance exchange adapter implementing IExchangeAdapter."""
from decimal import Decimal
import pandas as pd

from app.services.exchange.adapters.base import IExchangeAdapter
from app.services.exchange.binance_client import binance_client


class BinanceAdapter(IExchangeAdapter):
    """Adapter wrapping the existing BinanceClientWrapper."""

    def __init__(self):
        self._client = binance_client

    async def connect(self) -> None:
        await self._client.connect()

    async def disconnect(self) -> None:
        await self._client.disconnect()

    async def get_ticker_price(self, symbol: str) -> Decimal:
        return await self._client.get_ticker_price(symbol)

    async def get_klines(self, symbol: str, interval: str = "1h", limit: int = 500) -> pd.DataFrame:
        return await self._client.get_klines(symbol, interval, limit)

    async def get_balance(self, asset: str = "USDT") -> Decimal:
        return await self._client.get_balance(asset)

    async def place_market_order(self, symbol: str, side: str, quantity: float) -> dict:
        return await self._client.place_market_order(symbol, side, quantity)

    async def cancel_order(self, symbol: str, order_id: int) -> dict:
        return await self._client.cancel_order(symbol, order_id)

    async def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        return await self._client.get_open_orders(symbol)

    @property
    def name(self) -> str:
        return "binance"

    @property
    def is_connected(self) -> bool:
        return self._client._client is not None
