"""Trading journal model for trade notes and reflection."""
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Text, DateTime
from app.models.base import Base


class JournalEntry(Base):
    __tablename__ = "trade_journal"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(Integer, nullable=True, index=True)
    notes = Column(Text, nullable=False)
    tags = Column(Text, default="")  # Comma-separated tags
    sentiment = Column(String(20), default="neutral")  # bullish, bearish, neutral
    lessons_learned = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
