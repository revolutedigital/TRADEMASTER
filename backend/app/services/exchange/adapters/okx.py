"""OKX exchange adapter (stub - ready for implementation).

All live methods raise NotImplementedError until OKX API is integrated.
"""

from decimal import Decimal
from typing import Any

import pandas as pd

from app.core.logging import get_logger
from app.services.exchange.adapters.base import ExchangeAdapter

logger = get_logger(__name__)


class OKXAdapter(ExchangeAdapter):
    """OKX exchange adapter. Stub implementation for future integration."""

    def __init__(self) -> None:
        self._connected = False
        logger.info("okx_adapter_initialized", status="stub")

    async def connect(self) -> None:
        self._connected = True
        logger.info("okx_adapter_connected", mode="stub")

    async def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def get_ticker_price(self, symbol: str) -> Decimal:
        raise NotImplementedError("OKX adapter not yet implemented")

    async def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 500,
    ) -> pd.DataFrame:
        raise NotImplementedError("OKX adapter not yet implemented")

    async def get_balance(self, asset: str = "USDT") -> Decimal:
        raise NotImplementedError("OKX adapter not yet implemented")

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: float | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError("OKX adapter not yet implemented")

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
    ) -> dict[str, Any]:
        raise NotImplementedError("OKX adapter not yet implemented")

    async def cancel_order(self, symbol: str, order_id: str | int) -> dict[str, Any]:
        raise NotImplementedError("OKX adapter not yet implemented")

    async def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError("OKX adapter not yet implemented")

    @property
    def name(self) -> str:
        return "okx"
