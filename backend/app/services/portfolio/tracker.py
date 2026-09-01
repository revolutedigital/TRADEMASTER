"""Real-time portfolio position and P&L tracking."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.events import Event, EventType, event_bus
from app.core.logging import get_logger
from app.models.portfolio import PortfolioSnapshot, Position
from app.services.risk.stop_loss import stop_loss_calculator

logger = get_logger(__name__)


@dataclass(frozen=True)
class PaperExitCandidate:
    """A paper position that needs a reducing market order before ledger closure."""

    position: Position
    observed_price: float
    reason: Literal["STOP_LOSS", "TAKE_PROFIT", "TIME_EXIT"]


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
        execution_mode: str = "PAPER",
        entry_exchange_order_id: str | None = None,
        protective_order_list_id: int | None = None,
        protective_quantity: float | None = None,
        protection_status: str = "LOCAL",
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
            execution_mode=execution_mode,
            entry_exchange_order_id=entry_exchange_order_id,
            protective_order_list_id=protective_order_list_id,
            protective_quantity=protective_quantity,
            protection_status=protection_status,
            protection_updated_at=datetime.now(UTC),
            is_open=True,
            opened_at=datetime.now(UTC),
        )
        db.add(position)
        await db.flush()

        await event_bus.publish(
            Event(
                type=EventType.POSITION_OPENED,
                data={
                    "position_id": position.id,
                    "symbol": symbol,
                    "side": side,
                    "entry_price": entry_price,
                    "quantity": quantity,
                    "stop_loss": stop_loss_price,
                    "take_profit": take_profit_price,
                    "execution_mode": execution_mode,
                    "protection_status": protection_status,
                },
            )
        )

        logger.info(
            "position_opened",
            id=position.id,
            symbol=symbol,
            side=side,
            entry=entry_price,
            qty=quantity,
            execution_mode=execution_mode,
            protection_status=protection_status,
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
        position.realized_pnl = float(position.realized_pnl or 0) + pnl
        position.unrealized_pnl = 0
        position.closed_at = datetime.now(UTC)
        await db.flush()

        await event_bus.publish(
            Event(
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
            )
        )

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
        execution_mode: str | None = None,
    ) -> list[Position]:
        """Update prices for open positions, optionally scoped to one execution mode."""
        query = select(Position).where(Position.is_open.is_(True))
        if execution_mode is not None:
            query = query.where(Position.execution_mode == execution_mode)
        result = await db.execute(query)
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

    async def find_paper_exit_candidates(
        self,
        db: AsyncSession,
        prices: dict[str, float],
    ) -> list[PaperExitCandidate]:
        """Identify and lock paper exits without falsely closing the ledger first."""
        result = await db.execute(
            select(Position).where(
                Position.is_open.is_(True),
                Position.execution_mode == "PAPER",
            ).with_for_update()
        )
        positions = list(result.scalars().all())
        candidates: list[PaperExitCandidate] = []

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
                    candidates.append(PaperExitCandidate(pos, price, "STOP_LOSS"))
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
                    candidates.append(PaperExitCandidate(pos, price, "TAKE_PROFIT"))
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

            # Time-based exit: close ALL positions after max hold time
            # Both winners and losers — forces discipline, prevents stale positions
            if stop_loss_calculator.should_time_exit(pos.opened_at):
                logger.info(
                    "time_exit_triggered",
                    position_id=pos.id,
                    symbol=pos.symbol,
                    observed_price=price,
                )
                candidates.append(PaperExitCandidate(pos, price, "TIME_EXIT"))

        return candidates

    async def check_stop_losses(
        self,
        db: AsyncSession,
        prices: dict[str, float],
        execution_mode: str = "PAPER",
    ) -> list[PaperExitCandidate]:
        """Compatibility alias that only identifies paper exits, never closes them.

        A caller must execute and confirm the reducing market order before it
        calls ``close_position``. Keeping this alias detection-only prevents a
        future caller from accidentally bypassing the fill-confirmation rule.
        """
        if execution_mode != "PAPER":
            return []
        return await self.find_paper_exit_candidates(db, prices)

    async def get_open_positions(
        self,
        db: AsyncSession,
        symbol: str | None = None,
        execution_mode: str | None = None,
    ) -> list[Position]:
        """Get open positions, optionally scoped by symbol and execution mode."""
        query = select(Position).where(Position.is_open.is_(True))
        if symbol:
            query = query.where(Position.symbol == symbol)
        if execution_mode is not None:
            query = query.where(Position.execution_mode == execution_mode)
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_total_exposure(
        self,
        db: AsyncSession,
        execution_mode: str | None = None,
    ) -> float:
        """Get total notional value of open positions in an optional ledger."""
        positions = await self.get_open_positions(db, execution_mode=execution_mode)
        return sum(float(p.current_price) * float(p.quantity) for p in positions)

    async def get_symbol_exposure(
        self,
        db: AsyncSession,
        symbol: str,
        execution_mode: str | None = None,
    ) -> float:
        """Get symbol notional for one optional execution-mode ledger."""
        positions = await self.get_open_positions(
            db,
            symbol=symbol,
            execution_mode=execution_mode,
        )
        return sum(float(p.current_price) * float(p.quantity) for p in positions)

    async def take_snapshot(
        self, db: AsyncSession, equity: float, balance: float
    ) -> PortfolioSnapshot:
        """Record a portfolio snapshot for equity curve tracking."""
        positions = await self.get_open_positions(db)
        unrealized = sum(float(p.unrealized_pnl) for p in positions)

        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(UTC),
            total_equity=equity,
            available_balance=balance,
            unrealized_pnl=unrealized,
            open_positions_count=len(positions),
        )
        db.add(snapshot)
        await db.flush()
        return snapshot


portfolio_tracker = PortfolioTracker()
