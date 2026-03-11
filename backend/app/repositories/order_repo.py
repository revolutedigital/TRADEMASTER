"""Order repository: data access for orders and trades."""

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.trade import Order, OrderStatus
from app.repositories.base import BaseRepository


class OrderRepository(BaseRepository[Order]):
    def __init__(self) -> None:
        super().__init__(Order)

    async def list_filtered(
        self,
        db: AsyncSession,
        symbol: str | None = None,
        side: str | None = None,
        status: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Order]:
        """List orders with optional filters, ordered by newest first."""
        query = select(Order)
        if symbol and symbol.upper() != "ALL":
            query = query.where(Order.symbol == symbol.upper())
        if side and side.upper() != "ALL":
            query = query.where(Order.side == side.upper())
        if status and status.upper() != "ALL":
            query = query.where(Order.status == status.upper())
        if start_date:
            try:
                dt = datetime.fromisoformat(start_date)
                query = query.where(Order.created_at >= dt)
            except ValueError:
                pass
        if end_date:
            try:
                dt = datetime.fromisoformat(end_date)
                query = query.where(Order.created_at <= dt)
            except ValueError:
                pass
        query = query.order_by(Order.created_at.desc()).offset(offset).limit(limit)
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_open_orders(self, db: AsyncSession, symbol: str | None = None) -> list[Order]:
        query = select(Order).where(
            Order.status.in_([OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED])
        )
        if symbol:
            query = query.where(Order.symbol == symbol)
        result = await db.execute(query.order_by(Order.created_at.desc()))
        return list(result.scalars().all())

    async def get_by_exchange_id(self, db: AsyncSession, exchange_order_id: str) -> Order | None:
        result = await db.execute(
            select(Order).where(Order.exchange_order_id == exchange_order_id)
        )
        return result.scalar_one_or_none()

    async def get_filled_orders(
        self, db: AsyncSession, symbol: str | None = None, limit: int = 50
    ) -> list[Order]:
        query = select(Order).where(Order.status == OrderStatus.FILLED)
        if symbol:
            query = query.where(Order.symbol == symbol)
        result = await db.execute(query.order_by(Order.created_at.desc()).limit(limit))
        return list(result.scalars().all())

    async def get_orders_by_symbol(
        self, db: AsyncSession, symbol: str, limit: int = 50
    ) -> list[Order]:
        result = await db.execute(
            select(Order)
            .where(Order.symbol == symbol)
            .order_by(Order.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
