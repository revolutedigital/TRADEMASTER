"""Exchange adapter factory for multi-exchange support.

Usage:
    from app.services.exchange.factory import get_exchange_adapter

    adapter = get_exchange_adapter("binance")
    price = await adapter.get_ticker_price("BTCUSDT")

    # List available exchanges
    exchanges = list_exchanges()  # ["binance", "bybit", "okx"]
"""

from app.core.logging import get_logger
from app.services.exchange.adapters.base import ExchangeAdapter
from app.services.exchange.adapters.binance_adapter import BinanceAdapter
from app.services.exchange.adapters.bybit_adapter import BybitAdapter
from app.services.exchange.adapters.okx import OKXAdapter

logger = get_logger(__name__)

_ADAPTERS: dict[str, type[ExchangeAdapter]] = {
    "binance": BinanceAdapter,
    "bybit": BybitAdapter,
    "okx": OKXAdapter,
}

# Cache adapter instances to avoid creating multiple wrappers around same client
_instances: dict[str, ExchangeAdapter] = {}


def get_exchange_adapter(
    exchange: str = "binance",
    *,
    cached: bool = True,
) -> ExchangeAdapter:
    """Factory: instantiate an exchange adapter by name.

    Args:
        exchange: Exchange identifier ("binance", "bybit", "okx").
        cached: If True, reuse existing adapter instance (default).
                Set to False for testing or independent instances.

    Returns:
        ExchangeAdapter instance for the requested exchange.

    Raises:
        ValueError: If the exchange name is not recognized.
    """
    key = exchange.lower()
    adapter_cls = _ADAPTERS.get(key)
    if not adapter_cls:
        available = ", ".join(sorted(_ADAPTERS.keys()))
        raise ValueError(f"Unknown exchange: {exchange}. Available: {available}")

    if cached and key in _instances:
        return _instances[key]

    adapter = adapter_cls()
    if cached:
        _instances[key] = adapter

    logger.info("exchange_adapter_created", exchange=key, adapter=adapter_cls.__name__)
    return adapter


def list_exchanges() -> list[str]:
    """Return list of available exchange adapter names."""
    return sorted(_ADAPTERS.keys())


def register_adapter(name: str, adapter_cls: type[ExchangeAdapter]) -> None:
    """Register a custom exchange adapter at runtime.

    Useful for plugins or testing.

    Args:
        name: Short identifier for the exchange.
        adapter_cls: Class that implements ExchangeAdapter.
    """
    _ADAPTERS[name.lower()] = adapter_cls
    logger.info("exchange_adapter_registered", exchange=name, adapter=adapter_cls.__name__)
