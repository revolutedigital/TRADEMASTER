"""Market making bot with automated spread management and inventory risk control."""

import asyncio
from dataclasses import dataclass
from decimal import Decimal

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MarketMakerConfig:
    symbol: str
    spread_bps: int = 10  # Spread in basis points
    order_size: Decimal = Decimal("0.01")
    max_inventory: Decimal = Decimal("1.0")
    inventory_skew_factor: float = 0.5
    refresh_interval_seconds: float = 5.0
    max_open_orders: int = 5


class MarketMaker:
    """Automated market making with inventory risk management."""

    def __init__(self):
        self._running = False
        self._config: MarketMakerConfig | None = None
        self._current_inventory: Decimal = Decimal("0")
        self._open_orders: list[dict] = []
        self._total_volume: Decimal = Decimal("0")
        self._total_pnl: Decimal = Decimal("0")

    async def start(self, config: MarketMakerConfig):
        """Start market making."""
        self._config = config
        self._running = True
        logger.info("market_maker_started", symbol=config.symbol, spread=config.spread_bps)

    async def stop(self):
        """Stop market making and cancel all orders."""
        self._running = False
        self._open_orders.clear()
        logger.info("market_maker_stopped")

    def calculate_quotes(self, mid_price: Decimal) -> tuple[Decimal, Decimal]:
        """Calculate bid/ask prices with inventory skew."""
        if not self._config:
            return mid_price, mid_price

        half_spread = mid_price * Decimal(str(self._config.spread_bps)) / Decimal("20000")
        
        # Skew prices based on inventory to reduce risk
        inventory_ratio = float(self._current_inventory / self._config.max_inventory) if self._config.max_inventory > 0 else 0
        skew = Decimal(str(inventory_ratio * self._config.inventory_skew_factor)) * half_spread
        
        bid = mid_price - half_spread - skew  # Lower bid when long inventory
        ask = mid_price + half_spread - skew  # Lower ask when long inventory
        
        return bid.quantize(Decimal("0.01")), ask.quantize(Decimal("0.01"))

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "symbol": self._config.symbol if self._config else None,
            "current_inventory": str(self._current_inventory),
            "open_orders": len(self._open_orders),
            "total_volume": str(self._total_volume),
            "total_pnl": str(self._total_pnl),
        }


market_maker = MarketMaker()
