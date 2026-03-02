"""Position repository: data access for positions and portfolio snapshots."""

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.portfolio import Position, PortfolioSnapshot
from app.repositories.base import BaseRepository


class PositionRepository(BaseRepository[Position]):
    def __init__(self) -> None:
        super().__init__(Position)

    async def get_open(self, db: AsyncSession, symbol: str | None = None) -> list[Position]:
        query = select(Position).where(Position.is_open == True)
        if symbol:
            query = query.where(Position.symbol == symbol)
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_closed(
        self, db: AsyncSession, symbol: str | None = None, limit: int = 50
    ) -> list[Position]:
        query = select(Position).where(Position.is_open == False)
        if symbol:
            query = query.where(Position.symbol == symbol)
        result = await db.execute(query.order_by(Position.closed_at.desc()).limit(limit))
        return list(result.scalars().all())

    async def get_by_symbol_and_side(
        self, db: AsyncSession, symbol: str, side: str
    ) -> list[Position]:
        result = await db.execute(
            select(Position).where(
                Position.symbol == symbol,
                Position.side == side,
                Position.is_open == True,
            )
        )
        return list(result.scalars().all())

    async def get_total_exposure(self, db: AsyncSession) -> float:
        positions = await self.get_open(db)
        return sum(float(p.current_price) * float(p.quantity) for p in positions)

    async def get_symbol_exposure(self, db: AsyncSession, symbol: str) -> float:
        positions = await self.get_open(db, symbol)
        return sum(float(p.current_price) * float(p.quantity) for p in positions)


class SnapshotRepository(BaseRepository[PortfolioSnapshot]):
    def __init__(self) -> None:
        super().__init__(PortfolioSnapshot)

    async def get_recent(
        self, db: AsyncSession, limit: int = 100
    ) -> list[PortfolioSnapshot]:
        result = await db.execute(
            select(PortfolioSnapshot)
            .order_by(PortfolioSnapshot.timestamp.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_since(
        self, db: AsyncSession, since: datetime
    ) -> list[PortfolioSnapshot]:
        result = await db.execute(
            select(PortfolioSnapshot)
            .where(PortfolioSnapshot.timestamp >= since)
            .order_by(PortfolioSnapshot.timestamp.asc())
        )
        return list(result.scalars().all())
