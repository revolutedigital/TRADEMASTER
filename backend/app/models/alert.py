"""Price alert models."""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text
from app.models.base import Base, TimestampMixin


class PriceAlert(Base, TimestampMixin):
    __tablename__ = "price_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    condition = Column(String(10), nullable=False)  # "above" or "below"
    target_price = Column(Float, nullable=False)
    is_triggered = Column(Boolean, default=False, nullable=False)
    triggered_at = Column(DateTime(timezone=True), nullable=True)
    notes = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
