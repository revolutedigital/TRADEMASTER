"""Smart Order Router: find best execution venue."""
from dataclasses import dataclass
from decimal import Decimal

from app.core.logging import get_logger
from app.services.exchange.factory import get_exchange_adapter, list_exchanges

logger = get_logger(__name__)


@dataclass
class VenueQuote:
    exchange: str
    price: Decimal
    available: bool


class SmartOrderRouter:
    """Compare prices across exchanges and route to best venue."""

    async def find_best_venue(self, symbol: str, side: str) -> str:
        """Find the exchange with the best price for the given order."""
        quotes: list[VenueQuote] = []

        for exchange_name in list_exchanges():
            try:
                adapter = get_exchange_adapter(exchange_name)
                if not adapter.is_connected:
                    continue
                price = await adapter.get_ticker_price(symbol)
                quotes.append(VenueQuote(exchange=exchange_name, price=price, available=True))
            except Exception:
                quotes.append(VenueQuote(exchange=exchange_name, price=Decimal("0"), available=False))

        available = [q for q in quotes if q.available]
        if not available:
            return "binance"  # Default fallback

        if side == "BUY":
            best = min(available, key=lambda q: q.price)
        else:
            best = max(available, key=lambda q: q.price)

        logger.info("smart_route_selected", exchange=best.exchange, price=float(best.price), side=side)
        return best.exchange


smart_order_router = SmartOrderRouter()
