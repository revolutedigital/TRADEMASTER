"""Order lifecycle management: signal to execution.

Supports both live (Binance) and paper trading modes.
Paper mode simulates fills with realistic slippage and commission.

All financial calculations use Decimal for precision — IEEE 754 float
rounding errors are unacceptable in a trading system.
"""

import random
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.events import Event, EventType, event_bus
from app.core.exceptions import OrderExecutionError
from app.core.logging import get_logger
from app.models.trade import Order, OrderSide, OrderStatus, OrderType
from app.services.exchange.binance_client import binance_client

logger = get_logger(__name__)

# Paper trading simulation parameters (Decimal for precision)
PAPER_COMMISSION_RATE = Decimal("0.001")  # 0.1% taker fee (Binance standard)
PAPER_SLIPPAGE_BPS = 5  # 5 basis points max (0.05%) slippage


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
        """Simulate a market order with realistic slippage and fees.

        All price/commission arithmetic uses Decimal to avoid float rounding
        errors that can accumulate over many trades.
        """
        qty = Decimal(str(quantity))

        # Get current price from Binance (already returns Decimal)
        try:
            current_price = await binance_client.get_ticker_price(symbol)
        except Exception:
            # Fallback: get from DB
            from sqlalchemy import select
            from app.models.market import OHLCV
            result = await db.execute(
                select(OHLCV)
                .where(OHLCV.symbol == symbol, OHLCV.interval == "15m")
                .order_by(OHLCV.open_time.desc())
                .limit(1)
            )
            candle = result.scalar_one_or_none()
            if not candle:
                raise OrderExecutionError(f"No price data for {symbol}")
            current_price = Decimal(str(candle.close))

        # Apply bilateral slippage: 50% chance favorable, 50% adverse (realistic)
        slippage_bps = Decimal(str(random.uniform(0, PAPER_SLIPPAGE_BPS)))
        slippage_pct = slippage_bps / Decimal("10000")
        slippage_direction = random.choice([-1, 1])  # -1 = favorable, +1 = adverse
        if side == "BUY":
            fill_price = current_price * (Decimal("1") + slippage_pct * slippage_direction)
        else:
            fill_price = current_price * (Decimal("1") - slippage_pct * slippage_direction)

        # Round to 2 decimal places (USDT precision)
        fill_price = fill_price.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        commission = (fill_price * qty * PAPER_COMMISSION_RATE).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )

        order = Order(
            exchange_order_id=f"PAPER-{int(datetime.now(timezone.utc).timestamp() * 1000)}",
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            quantity=float(qty),
            price=float(fill_price),
            filled_quantity=float(qty),
            avg_fill_price=float(fill_price),
            commission=float(commission),
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
                "quantity": float(qty),
                "avg_price": float(fill_price),
                "commission": float(commission),
                "paper_mode": True,
            },
        ))

        logger.info(
            "paper_order_executed",
            order_id=order.id,
            symbol=symbol,
            side=side,
            filled_qty=float(qty),
            fill_price=float(fill_price),
            slippage_bps=float(slippage_bps),
            commission=float(commission),
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
