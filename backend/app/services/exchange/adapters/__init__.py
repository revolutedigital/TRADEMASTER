"""Exchange adapters for multi-exchange support.

Available adapters:
- BinanceAdapter: Production-ready, wraps existing BinanceClientWrapper
- BybitAdapter: Paper trading ready, live methods pending
- OKXAdapter: Stub for future integration

Usage:
    from app.services.exchange.factory import get_exchange_adapter
    adapter = get_exchange_adapter("binance")
"""

from app.services.exchange.adapters.base import ExchangeAdapter, IExchangeAdapter
from app.services.exchange.adapters.binance_adapter import BinanceAdapter
from app.services.exchange.adapters.bybit_adapter import BybitAdapter
from app.services.exchange.adapters.okx import OKXAdapter

__all__ = [
    "ExchangeAdapter",
    "IExchangeAdapter",
    "BinanceAdapter",
    "BybitAdapter",
    "OKXAdapter",
]
