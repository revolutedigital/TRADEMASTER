"""Trading journal API endpoints."""
import json

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
    tags: list[str] | str = ""
    sentiment: str = "neutral"
    lessons_learned: str | None = None


def _tags_to_str(tags: list[str] | str) -> str:
    """Convert tags input (array or string) to comma-separated string for DB."""
    if isinstance(tags, list):
        return ",".join(t.strip() for t in tags if t.strip())
    return tags


def _tags_to_list(tags_str: str | None) -> list[str]:
    """Convert comma-separated tags string to array for API response."""
    if not tags_str:
        return []
    return [t.strip() for t in tags_str.split(",") if t.strip()]


@router.post("")
async def create_entry(body: JournalRequest, db: AsyncSession = Depends(get_db), _auth: dict = Depends(require_auth)):
    data = body.model_dump()
    data["tags"] = _tags_to_str(data["tags"])
    entry = JournalEntry(**data)
    db.add(entry)
    await db.commit()
    return {"id": entry.id, "notes": entry.notes, "sentiment": entry.sentiment, "tags": _tags_to_list(entry.tags)}


@router.get("")
async def list_entries(limit: int = 50, db: AsyncSession = Depends(get_db), _auth: dict = Depends(require_auth)):
    result = await db.execute(select(JournalEntry).order_by(JournalEntry.created_at.desc()).limit(limit))
    entries = result.scalars().all()
    return [
        {
            "id": e.id,
            "trade_id": e.trade_id,
            "notes": e.notes,
            "tags": _tags_to_list(e.tags),
            "sentiment": e.sentiment,
            "lessons_learned": e.lessons_learned or "",
            "created_at": str(e.created_at),
        }
        for e in entries
    ]
