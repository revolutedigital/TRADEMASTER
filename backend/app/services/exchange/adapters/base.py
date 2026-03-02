"""Base exchange adapter interface."""
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any

import pandas as pd


class IExchangeAdapter(ABC):
    """Unified interface for all exchange integrations."""

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def get_ticker_price(self, symbol: str) -> Decimal: ...

    @abstractmethod
    async def get_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame: ...

    @abstractmethod
    async def get_balance(self, asset: str) -> Decimal: ...

    @abstractmethod
    async def place_market_order(self, symbol: str, side: str, quantity: float) -> dict: ...

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: int) -> dict: ...

    @abstractmethod
    async def get_open_orders(self, symbol: str | None = None) -> list[dict]: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...
