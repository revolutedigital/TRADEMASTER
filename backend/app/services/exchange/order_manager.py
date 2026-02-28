"""Order lifecycle management: signal to execution.

Supports both live (Binance) and paper trading modes.
Paper mode simulates fills with realistic slippage and commission.
"""

import random
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.events import Event, EventType, event_bus
from app.core.exceptions import OrderExecutionError
from app.core.logging import get_logger
from app.models.trade import Order, OrderSide, OrderStatus, OrderType
from app.services.exchange.binance_client import binance_client

logger = get_logger(__name__)

# Paper trading simulation parameters
PAPER_COMMISSION_RATE = 0.001  # 0.1% taker fee (Binance standard)
PAPER_SLIPPAGE_BPS = 5  # 5 basis points (0.05%) slippage


class OrderManager:
    """Manages the full order lifecycle from signal to fill."""

    async def execute_market_order(
        self,
        db: AsyncSession,
        symbol: str,
        side: str,
        quantity: float,
        signal_id: int | None = None,
    ) -> Order:
        """Execute a market order. Routes to paper or live mode based on config."""
        if settings.paper_mode:
            return await self._execute_paper_order(db, symbol, side, quantity, signal_id)
        return await self._execute_live_order(db, symbol, side, quantity, signal_id)

    async def _execute_paper_order(
        self,
        db: AsyncSession,
        symbol: str,
        side: str,
        quantity: float,
        signal_id: int | None = None,
    ) -> Order:
        """Simulate a market order with realistic slippage and fees."""
        # Get current price from Binance (read-only, no trade execution)
        try:
            current_price = float(await binance_client.get_ticker_price(symbol))
        except Exception:
            # Fallback: get from DB
            from sqlalchemy import select
            from app.models.market import OHLCV
            result = await db.execute(
                select(OHLCV)
                .where(OHLCV.symbol == symbol, OHLCV.interval == "1m")
                .order_by(OHLCV.open_time.desc())
                .limit(1)
            )
            candle = result.scalar_one_or_none()
            if not candle:
                raise OrderExecutionError(f"No price data for {symbol}")
            current_price = float(candle.close)

        # Apply slippage (adverse direction)
        slippage_pct = random.uniform(0, PAPER_SLIPPAGE_BPS) / 10000
        if side == "BUY":
            fill_price = current_price * (1 + slippage_pct)
        else:
            fill_price = current_price * (1 - slippage_pct)

        commission = fill_price * quantity * PAPER_COMMISSION_RATE

        order = Order(
            exchange_order_id=f"PAPER-{int(datetime.now(timezone.utc).timestamp() * 1000)}",
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            quantity=quantity,
            price=fill_price,
            filled_quantity=quantity,
            avg_fill_price=fill_price,
            commission=commission,
            signal_id=signal_id,
            notes="Paper trade (simulated with slippage)",
        )
        db.add(order)
        await db.flush()

        # Publish event
        await event_bus.publish(Event(
            type=EventType.ORDER_FILLED,
            data={
                "order_id": order.id,
                "exchange_order_id": order.exchange_order_id,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "avg_price": fill_price,
                "commission": commission,
                "paper_mode": True,
            },
        ))

        logger.info(
            "paper_order_executed",
            order_id=order.id,
            symbol=symbol,
            side=side,
            filled_qty=quantity,
            fill_price=round(fill_price, 2),
            slippage_bps=round(slippage_pct * 10000, 2),
            commission=round(commission, 4),
        )
        return order

    async def _execute_live_order(
        self,
        db: AsyncSession,
        symbol: str,
        side: str,
        quantity: float,
        signal_id: int | None = None,
    ) -> Order:
        """Execute a real market order on Binance."""
        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            quantity=quantity,
            signal_id=signal_id,
        )
        db.add(order)
        await db.flush()

        try:
            result = await binance_client.place_market_order(symbol, side, quantity)

            order.exchange_order_id = str(result["orderId"])
            order.status = OrderStatus(result.get("status", "FILLED"))
            order.filled_quantity = float(result.get("executedQty", 0))
            order.avg_fill_price = float(result.get("avgPrice", 0)) or None

            commission = sum(
                float(fill.get("commission", 0))
                for fill in result.get("fills", [])
            )
            order.commission = commission

            await db.flush()

            await event_bus.publish(Event(
                type=EventType.ORDER_FILLED,
                data={
                    "order_id": order.id,
                    "exchange_order_id": order.exchange_order_id,
                    "symbol": symbol,
                    "side": side,
                    "quantity": order.filled_quantity,
                    "avg_price": order.avg_fill_price,
                    "commission": commission,
                    "paper_mode": False,
                },
            ))

            logger.info(
                "live_order_executed",
                order_id=order.id,
                symbol=symbol,
                side=side,
                filled_qty=order.filled_quantity,
                avg_price=order.avg_fill_price,
            )
            return order

        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.notes = str(e)[:500]
            await db.flush()
            logger.error("order_execution_failed", order_id=order.id, error=str(e))
            raise OrderExecutionError(f"Order failed: {e}") from e

    async def cancel_order(self, db: AsyncSession, order: Order) -> Order:
        """Cancel an open order."""
        if not settings.paper_mode and order.exchange_order_id:
            await binance_client.cancel_order(order.symbol, int(order.exchange_order_id))

        order.status = OrderStatus.CANCELLED
        await db.flush()

        await event_bus.publish(Event(
            type=EventType.ORDER_CANCELLED,
            data={"order_id": order.id, "symbol": order.symbol},
        ))
        return order


order_manager = OrderManager()
