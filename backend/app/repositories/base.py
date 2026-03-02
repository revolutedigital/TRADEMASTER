"""Base repository with generic CRUD operations."""

from typing import Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import Base

T = TypeVar("T", bound=Base)


class BaseRepository(Generic[T]):
    """Generic repository providing basic CRUD over SQLAlchemy models."""

    def __init__(self, model_class: type[T]) -> None:
        self._model = model_class

    async def get_by_id(self, db: AsyncSession, entity_id: int) -> T | None:
        result = await db.execute(select(self._model).where(self._model.id == entity_id))
        return result.scalar_one_or_none()

    async def create(self, db: AsyncSession, entity: T) -> T:
        db.add(entity)
        await db.flush()
        return entity

    async def update(self, db: AsyncSession, entity: T) -> T:
        await db.flush()
        return entity

    async def delete(self, db: AsyncSession, entity: T) -> None:
        await db.delete(entity)
        await db.flush()

    async def list_all(self, db: AsyncSession, limit: int = 100) -> list[T]:
        result = await db.execute(select(self._model).limit(limit))
        return list(result.scalars().all())
