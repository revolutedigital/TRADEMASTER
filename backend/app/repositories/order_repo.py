"""Order repository: data access for orders and trades."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.trade import Order, OrderStatus
from app.repositories.base import BaseRepository


class OrderRepository(BaseRepository[Order]):
    def __init__(self) -> None:
        super().__init__(Order)

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
