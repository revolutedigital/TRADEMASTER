"""Custom order types: Iceberg, Bracket, OCO, Trailing Stop.

Implements complex order types that may not be natively supported by exchanges.
"""

import asyncio
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from app.core.logging import get_logger

logger = get_logger(__name__)


class CustomOrderType(str, Enum):
    ICEBERG = "iceberg"
    BRACKET = "bracket"
    OCO = "oco"  # One-Cancels-Other
    TRAILING_STOP = "trailing_stop"


@dataclass
class IcebergOrder:
    """Iceberg order: large order split into visible slices."""
    symbol: str
    side: str
    total_quantity: Decimal
    visible_quantity: Decimal
    price: Decimal
    filled_quantity: Decimal = Decimal("0")

    @property
    def remaining(self) -> Decimal:
        return self.total_quantity - self.filled_quantity

    @property
    def current_slice(self) -> Decimal:
        return min(self.visible_quantity, self.remaining)


@dataclass
class BracketOrder:
    """Bracket order: entry + take-profit + stop-loss."""
    symbol: str
    side: str
    quantity: Decimal
    entry_price: Decimal
    take_profit_price: Decimal
    stop_loss_price: Decimal
    status: str = "pending"  # pending, active, completed, cancelled


@dataclass
class OCOOrder:
    """One-Cancels-Other: two orders where filling one cancels the other."""
    symbol: str
    order_a: dict  # First order config
    order_b: dict  # Second order config
    triggered_order: str | None = None


@dataclass
class TrailingStopOrder:
    """Trailing stop: stop price follows market by a fixed distance."""
    symbol: str
    side: str
    quantity: Decimal
    trail_amount: Decimal | None = None  # Fixed dollar amount
    trail_pct: float | None = None  # Percentage trail
    activation_price: Decimal | None = None
    current_stop: Decimal | None = None
    highest_price: Decimal = Decimal("0")
    lowest_price: Decimal = Decimal("999999999")

    def update_price(self, current_price: Decimal) -> bool:
        """Update trailing stop with new price. Returns True if stop triggered."""
        if self.side == "SELL":
            # Long position: trail below market
            if current_price > self.highest_price:
                self.highest_price = current_price
                if self.trail_pct:
                    self.current_stop = current_price * (Decimal("1") - Decimal(str(self.trail_pct / 100)))
                elif self.trail_amount:
                    self.current_stop = current_price - self.trail_amount
            
            if self.current_stop and current_price <= self.current_stop:
                return True  # Triggered
        else:
            # Short position: trail above market
            if current_price < self.lowest_price:
                self.lowest_price = current_price
                if self.trail_pct:
                    self.current_stop = current_price * (Decimal("1") + Decimal(str(self.trail_pct / 100)))
                elif self.trail_amount:
                    self.current_stop = current_price + self.trail_amount
            
            if self.current_stop and current_price >= self.current_stop:
                return True  # Triggered
        
        return False


class CustomOrderManager:
    """Manage custom order types."""

    def __init__(self):
        self._iceberg_orders: dict[str, IcebergOrder] = {}
        self._bracket_orders: dict[str, BracketOrder] = {}
        self._oco_orders: dict[str, OCOOrder] = {}
        self._trailing_stops: dict[str, TrailingStopOrder] = {}

    async def create_iceberg(self, symbol: str, side: str, total_qty: Decimal, visible_qty: Decimal, price: Decimal) -> str:
        order_id = f"ice_{int(asyncio.get_event_loop().time() * 1000)}"
        self._iceberg_orders[order_id] = IcebergOrder(symbol=symbol, side=side, total_quantity=total_qty, visible_quantity=visible_qty, price=price)
        logger.info("iceberg_created", order_id=order_id, total=str(total_qty), visible=str(visible_qty))
        return order_id

    async def create_bracket(self, symbol: str, side: str, qty: Decimal, entry: Decimal, tp: Decimal, sl: Decimal) -> str:
        order_id = f"bkt_{int(asyncio.get_event_loop().time() * 1000)}"
        self._bracket_orders[order_id] = BracketOrder(symbol=symbol, side=side, quantity=qty, entry_price=entry, take_profit_price=tp, stop_loss_price=sl)
        logger.info("bracket_created", order_id=order_id, entry=str(entry), tp=str(tp), sl=str(sl))
        return order_id

    async def create_trailing_stop(self, symbol: str, side: str, qty: Decimal, trail_pct: float | None = None, trail_amount: Decimal | None = None) -> str:
        order_id = f"trl_{int(asyncio.get_event_loop().time() * 1000)}"
        self._trailing_stops[order_id] = TrailingStopOrder(symbol=symbol, side=side, quantity=qty, trail_pct=trail_pct, trail_amount=trail_amount)
        logger.info("trailing_stop_created", order_id=order_id, trail_pct=trail_pct, trail_amount=str(trail_amount) if trail_amount else None)
        return order_id

    def get_active_orders(self) -> dict:
        return {
            "iceberg": len(self._iceberg_orders),
            "bracket": len(self._bracket_orders),
            "oco": len(self._oco_orders),
            "trailing_stop": len(self._trailing_stops),
        }


custom_order_manager = CustomOrderManager()
