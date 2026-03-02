"""OKX exchange adapter (stub - ready for implementation)."""

from decimal import Decimal
from typing import Any

from app.core.logging import get_logger
from app.services.exchange.adapters.base import IExchangeAdapter

logger = get_logger(__name__)


class OKXAdapter(IExchangeAdapter):
    """OKX exchange adapter. Stub implementation for future integration."""

    def __init__(self):
        self._connected = False
        logger.info("okx_adapter_initialized", status="stub")

    async def connect(self) -> None:
        self._connected = True
        logger.info("okx_adapter_connected", mode="stub")

    async def disconnect(self) -> None:
        self._connected = False

    async def get_ticker_price(self, symbol: str) -> Decimal:
        raise NotImplementedError("OKX adapter not yet implemented")

    async def get_order_book(self, symbol: str, limit: int = 25) -> dict:
        raise NotImplementedError("OKX adapter not yet implemented")

    async def place_order(self, symbol: str, side: str, quantity: Decimal,
                          order_type: str = "MARKET", price: Decimal | None = None) -> dict:
        raise NotImplementedError("OKX adapter not yet implemented")

    async def cancel_order(self, symbol: str, order_id: str) -> dict:
        raise NotImplementedError("OKX adapter not yet implemented")

    async def get_balance(self, asset: str = "USDT") -> Decimal:
        raise NotImplementedError("OKX adapter not yet implemented")

    async def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        raise NotImplementedError("OKX adapter not yet implemented")

    @property
    def exchange_name(self) -> str:
        return "okx"

    @property
    def is_connected(self) -> bool:
        return self._connected
