"""Trading journal endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.dependencies import require_auth
from app.models.base import async_session_factory

router = APIRouter(prefix="/journal", tags=["journal"])


class JournalEntryRequest(BaseModel):
    trade_id: int | None = None
    title: str = Field(min_length=1, max_length=200)
    notes: str | None = None
    tags: list[str] = Field(default_factory=list)
    sentiment: str = Field(default="neutral", pattern="^(bullish|bearish|neutral)$")
    lessons_learned: str | None = None
    rating: int | None = Field(default=None, ge=1, le=5)


@router.post("")
async def create_journal_entry(req: JournalEntryRequest, _user: dict = Depends(require_auth)):
    from app.models.journal import JournalEntry
    async with async_session_factory() as db:
        entry = JournalEntry(
            trade_id=req.trade_id, title=req.title, notes=req.notes,
            tags=req.tags, sentiment=req.sentiment,
            lessons_learned=req.lessons_learned, rating=req.rating,
        )
        db.add(entry)
        await db.commit()
        await db.refresh(entry)
        return {"id": entry.id, "title": entry.title, "status": "created"}


@router.get("")
async def list_journal_entries(
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    _user: dict = Depends(require_auth),
):
    from sqlalchemy import select, func
    from app.models.journal import JournalEntry
    async with async_session_factory() as db:
        count_result = await db.execute(select(func.count(JournalEntry.id)))
        total = count_result.scalar() or 0

        result = await db.execute(
            select(JournalEntry)
            .order_by(JournalEntry.created_at.desc())
            .offset(offset).limit(limit)
        )
        entries = result.scalars().all()
        return {
            "total": total,
            "entries": [
                {"id": e.id, "trade_id": e.trade_id, "title": e.title,
                 "notes": e.notes, "tags": e.tags, "sentiment": e.sentiment,
                 "lessons_learned": e.lessons_learned, "rating": e.rating,
                 "created_at": e.created_at.isoformat() if e.created_at else None}
                for e in entries
            ],
        }
