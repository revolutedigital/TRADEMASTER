"""Trading journal API endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, require_auth
from app.models.journal import JournalEntry

router = APIRouter(prefix="/journal", tags=["journal"])


class JournalRequest(BaseModel):
    trade_id: int | None = None
    notes: str
    tags: str = ""
    sentiment: str = "neutral"
    lessons_learned: str | None = None


@router.post("")
async def create_entry(body: JournalRequest, db: AsyncSession = Depends(get_db), _auth: dict = Depends(require_auth)):
    entry = JournalEntry(**body.model_dump())
    db.add(entry)
    await db.commit()
    return {"id": entry.id, "notes": entry.notes, "sentiment": entry.sentiment}


@router.get("")
async def list_entries(limit: int = 50, db: AsyncSession = Depends(get_db), _auth: dict = Depends(require_auth)):
    result = await db.execute(select(JournalEntry).order_by(JournalEntry.created_at.desc()).limit(limit))
    entries = result.scalars().all()
    return [{"id": e.id, "trade_id": e.trade_id, "notes": e.notes, "tags": e.tags, "sentiment": e.sentiment, "lessons_learned": e.lessons_learned, "created_at": str(e.created_at)} for e in entries]
