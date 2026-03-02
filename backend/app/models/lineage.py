"""Data lineage tracking for TradeMaster pipeline."""

from sqlalchemy import Column, BigInteger, String, Text, DateTime, func
from app.models.base import Base


class DataLineage(Base):
    """Track data transformations through the pipeline."""

    __tablename__ = "data_lineage"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    source_type = Column(String(50), nullable=False)  # binance_api, computed, ml_output
    source_id = Column(String(200))
    destination_type = Column(String(50), nullable=False)
    destination_id = Column(String(200))
    transformation = Column(String(200))  # e.g. "feature_engineering", "prediction"
    metadata_ = Column("metadata", Text)  # JSON metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
