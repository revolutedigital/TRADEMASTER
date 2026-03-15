"""StoredEvent model for persistent event sourcing."""

from sqlalchemy import BigInteger, Column, DateTime, Integer, String, Text, func
from app.models.base import Base


class StoredEvent(Base):
    """Persistent storage for domain events (append-only)."""

    __tablename__ = "stored_events"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    event_id = Column(String(36), unique=True, nullable=False, index=True)
    event_type = Column(String(100), nullable=False, index=True)
    aggregate_type = Column(String(100), nullable=False, index=True)
    aggregate_id = Column(String(200), nullable=False, index=True)
    data = Column(Text, nullable=False)  # JSON serialized
    metadata_ = Column("metadata", Text, nullable=False, default="{}")  # JSON serialized
    version = Column(Integer, nullable=False)
    event_timestamp = Column(String(50), nullable=False)  # ISO timestamp from the event
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
