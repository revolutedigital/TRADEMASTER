"""Real-time portfolio position and P&L tracking."""

from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.events import Event, EventType, event_bus
from app.core.logging import get_logger
from app.models.portfolio import Position, PortfolioSnapshot
from app.services.risk.stop_loss import stop_loss_calculator

logger = get_logger(__name__)


class PortfolioTracker:
    """Tracks open positions, calculates P&L, and manages position lifecycle."""

    async def open_position(
        self,
        db: AsyncSession,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
    ) -> Position:
        """Record a new open position."""
        position = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            current_price=entry_price,
            unrealized_pnl=0,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            is_open=True,
            opened_at=datetime.now(timezone.utc),
        )
        db.add(position)
        await db.flush()

        await event_bus.publish(Event(
            type=EventType.POSITION_OPENED,
            data={
                "position_id": position.id,
                "symbol": symbol,
                "side": side,
                "entry_price": entry_price,
                "quantity": quantity,
                "stop_loss": stop_loss_price,
                "take_profit": take_profit_price,
            },
        ))

        logger.info(
            "position_opened",
            id=position.id,
            symbol=symbol,
            side=side,
            entry=entry_price,
            qty=quantity,
        )
        return position

    async def close_position(
        self,
        db: AsyncSession,
        position: Position,
        exit_price: float,
    ) -> Position:
        """Close a position and calculate realized P&L."""
        if position.side == "LONG":
            pnl = (exit_price - float(position.entry_price)) * float(position.quantity)
        else:
            pnl = (float(position.entry_price) - exit_price) * float(position.quantity)

        position.is_open = False
        position.current_price = exit_price
        position.realized_pnl = pnl
        position.unrealized_pnl = 0
        position.closed_at = datetime.now(timezone.utc)
        await db.flush()

        await event_bus.publish(Event(
            type=EventType.POSITION_CLOSED,
            data={
                "position_id": position.id,
                "symbol": position.symbol,
                "side": position.side,
                "entry_price": float(position.entry_price),
                "exit_price": exit_price,
                "pnl": pnl,
                "quantity": float(position.quantity),
            },
        ))

        logger.info(
            "position_closed",
            id=position.id,
            symbol=position.symbol,
            pnl=round(pnl, 2),
            exit=exit_price,
        )
        return position

    async def update_prices(
        self,
        db: AsyncSession,
        prices: dict[str, float],
    ) -> list[Position]:
        """Update current prices and unrealized P&L for all open positions."""
        result = await db.execute(
            select(Position).where(Position.is_open == True)
        )
        positions = list(result.scalars().all())

        for pos in positions:
            price = prices.get(pos.symbol)
            if price is None:
                continue

            pos.current_price = price
            if pos.side == "LONG":
                pos.unrealized_pnl = (price - float(pos.entry_price)) * float(pos.quantity)
            else:
                pos.unrealized_pnl = (float(pos.entry_price) - price) * float(pos.quantity)

        await db.flush()
        return positions

    async def check_stop_losses(
        self,
        db: AsyncSession,
        prices: dict[str, float],
    ) -> list[Position]:
        """Check if any open positions hit stop loss or take profit."""
        result = await db.execute(
            select(Position).where(Position.is_open == True)
        )
        positions = list(result.scalars().all())
        to_close = []

        for pos in positions:
            price = prices.get(pos.symbol)
            if price is None:
                continue

            # Check stop loss
            if pos.stop_loss_price:
                if stop_loss_calculator.is_stop_hit(price, float(pos.stop_loss_price), pos.side):
                    logger.warning(
                        "stop_loss_hit",
                        position_id=pos.id,
                        symbol=pos.symbol,
                        price=price,
                        stop=float(pos.stop_loss_price),
                    )
                    to_close.append((pos, price))
                    continue

            # Check take profit
            if pos.take_profit_price:
                if stop_loss_calculator.is_take_profit_hit(
                    price, float(pos.take_profit_price), pos.side
                ):
                    logger.info(
                        "take_profit_hit",
                        position_id=pos.id,
                        symbol=pos.symbol,
                        price=price,
                        tp=float(pos.take_profit_price),
                    )
                    to_close.append((pos, price))
                    continue

            # Update trailing stop
            if pos.stop_loss_price:
                new_stop = stop_loss_calculator.update_trailing_stop(
                    entry_price=float(pos.entry_price),
                    current_price=price,
                    current_stop=float(pos.stop_loss_price),
                    side=pos.side,
                )
                if new_stop != float(pos.stop_loss_price):
                    pos.stop_loss_price = new_stop
                    logger.info(
                        "trailing_stop_updated",
                        position_id=pos.id,
                        new_stop=round(new_stop, 2),
                    )

            # Check time-based exit
            if stop_loss_calculator.should_time_exit(pos.opened_at):
                # Only exit if position is not in profit
                if pos.side == "LONG":
                    in_profit = price > float(pos.entry_price)
                else:
                    in_profit = price < float(pos.entry_price)

                if not in_profit:
                    logger.info("time_exit_triggered", position_id=pos.id)
                    to_close.append((pos, price))

        # Close positions
        closed = []
        for pos, exit_price in to_close:
            closed_pos = await self.close_position(db, pos, exit_price)
            closed.append(closed_pos)

        return closed

    async def get_open_positions(self, db: AsyncSession, symbol: str | None = None) -> list[Position]:
        """Get all open positions, optionally filtered by symbol."""
        query = select(Position).where(Position.is_open == True)
        if symbol:
            query = query.where(Position.symbol == symbol)
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_total_exposure(self, db: AsyncSession) -> float:
        """Get total notional value of all open positions."""
        positions = await self.get_open_positions(db)
        return sum(
            float(p.current_price) * float(p.quantity)
            for p in positions
        )

    async def get_symbol_exposure(self, db: AsyncSession, symbol: str) -> float:
        """Get total notional for a specific symbol."""
        positions = await self.get_open_positions(db, symbol)
        return sum(
            float(p.current_price) * float(p.quantity)
            for p in positions
        )

    async def take_snapshot(self, db: AsyncSession, equity: float, balance: float) -> PortfolioSnapshot:
        """Record a portfolio snapshot for equity curve tracking."""
        positions = await self.get_open_positions(db)
        unrealized = sum(float(p.unrealized_pnl) for p in positions)

        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_equity=equity,
            available_balance=balance,
            unrealized_pnl=unrealized,
            open_positions_count=len(positions),
        )
        db.add(snapshot)
        await db.flush()
        return snapshot


portfolio_tracker = PortfolioTracker()
