"""Exchange adapter factory for multi-exchange support."""
from app.services.exchange.adapters.base import IExchangeAdapter
from app.services.exchange.adapters.binance_adapter import BinanceAdapter
from app.services.exchange.adapters.bybit import BybitAdapter
from app.core.logging import get_logger

logger = get_logger(__name__)

_ADAPTERS: dict[str, type[IExchangeAdapter]] = {
    "binance": BinanceAdapter,
    "bybit": BybitAdapter,
}


def get_exchange_adapter(exchange: str = "binance") -> IExchangeAdapter:
    """Factory: instantiate an exchange adapter by name."""
    adapter_cls = _ADAPTERS.get(exchange.lower())
    if not adapter_cls:
        raise ValueError(f"Unknown exchange: {exchange}. Available: {list(_ADAPTERS.keys())}")
    return adapter_cls()


def list_exchanges() -> list[str]:
    return list(_ADAPTERS.keys())
