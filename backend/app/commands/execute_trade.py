"""Command: Execute a trade order."""

from dataclasses import dataclass
from decimal import Decimal

from app.core.logging import get_logger
from app.services.exchange.order_manager import order_manager

logger = get_logger(__name__)


@dataclass
class ExecuteTradeCommand:
    symbol: str
    side: str  # BUY or SELL
    quantity: Decimal
    order_type: str = "MARKET"
    price: Decimal | None = None


class ExecuteTradeHandler:
    async def handle(self, cmd: ExecuteTradeCommand) -> dict:
        logger.info("cmd_execute_trade", symbol=cmd.symbol, side=cmd.side, qty=str(cmd.quantity))
        result = await order_manager.place_order(
            symbol=cmd.symbol,
            side=cmd.side,
            quantity=cmd.quantity,
            order_type=cmd.order_type,
            price=cmd.price,
        )
        return result


execute_trade_handler = ExecuteTradeHandler()
