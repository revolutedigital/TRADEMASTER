"""Trading journal models."""
from sqlalchemy import Column, Integer, String, Text, ForeignKey, JSON
from app.models.base import Base, TimestampMixin


class JournalEntry(Base, TimestampMixin):
    __tablename__ = "trade_journal"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(Integer, ForeignKey("trades.id", ondelete="SET NULL"), nullable=True)
    title = Column(String(200), nullable=False)
    notes = Column(Text, nullable=True)
    tags = Column(JSON, default=list)  # ["swing", "momentum", etc]
    sentiment = Column(String(20), default="neutral")  # bullish/bearish/neutral
    lessons_learned = Column(Text, nullable=True)
    rating = Column(Integer, nullable=True)  # 1-5 self-rating of the trade
