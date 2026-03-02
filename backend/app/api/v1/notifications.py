"""Notification API endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, require_auth
from app.models.notification import Notification

router = APIRouter()


class NotificationOut(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    type: str
    title: str
    message: str
    severity: str
    is_read: bool
    created_at: datetime


class NotificationCount(BaseModel):
    total: int
    unread: int


@router.get("/", response_model=list[NotificationOut])
async def list_notifications(
    limit: int = 50,
    unread_only: bool = False,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
):
    """List notifications, newest first."""
    query = select(Notification).order_by(Notification.created_at.desc()).limit(limit)
    if unread_only:
        query = query.where(Notification.is_read == False)  # noqa: E712
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/count", response_model=NotificationCount)
async def notification_count(
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
):
    """Get total and unread notification counts."""
    total_result = await db.execute(select(func.count(Notification.id)))
    total = total_result.scalar() or 0

    unread_result = await db.execute(
        select(func.count(Notification.id)).where(Notification.is_read == False)  # noqa: E712
    )
    unread = unread_result.scalar() or 0

    return NotificationCount(total=total, unread=unread)


@router.post("/{notification_id}/read")
async def mark_as_read(
    notification_id: int,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
):
    """Mark a notification as read."""
    await db.execute(
        update(Notification)
        .where(Notification.id == notification_id)
        .values(is_read=True, read_at=datetime.now(timezone.utc))
    )
    await db.commit()
    return {"status": "ok"}


@router.post("/read-all")
async def mark_all_read(
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
):
    """Mark all notifications as read."""
    await db.execute(
        update(Notification)
        .where(Notification.is_read == False)  # noqa: E712
        .values(is_read=True, read_at=datetime.now(timezone.utc))
    )
    await db.commit()
    return {"status": "ok"}
