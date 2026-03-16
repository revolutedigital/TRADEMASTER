"""Data lineage tracking models for TradeMaster pipeline.

Three tables:
- lineage_nodes: Every artifact in the DAG (data sources, features, models, etc.)
- lineage_edges: Directed dependencies between nodes
- lineage_audit_log: Append-only log of lineage operations
"""

from sqlalchemy import Column, BigInteger, String, Text, DateTime, func, UniqueConstraint
from app.models.base import Base


class DataLineage(Base):
    """Legacy table kept for backward compatibility."""

    __tablename__ = "data_lineage"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    source_type = Column(String(50), nullable=False)
    source_id = Column(String(200))
    destination_type = Column(String(50), nullable=False)
    destination_id = Column(String(200))
    transformation = Column(String(200))
    metadata_ = Column("metadata", Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class LineageNodeRecord(Base):
    """Persistent storage for lineage DAG nodes."""

    __tablename__ = "lineage_nodes"

    id = Column(String(200), primary_key=True)
    node_type = Column(String(50), nullable=False, index=True)
    name = Column(String(500), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    version = Column(String(50), nullable=False, server_default="1")
    metadata_ = Column("metadata", Text, server_default="{}")


class LineageEdgeRecord(Base):
    """Persistent storage for lineage DAG edges."""

    __tablename__ = "lineage_edges"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    source_id = Column(String(200), nullable=False, index=True)
    target_id = Column(String(200), nullable=False, index=True)
    edge_type = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    metadata_ = Column("metadata", Text, server_default="{}")

    __table_args__ = (
        UniqueConstraint("source_id", "target_id", "edge_type", name="uq_lineage_edge"),
    )


class LineageAuditLogRecord(Base):
    """Persistent storage for lineage audit trail."""

    __tablename__ = "lineage_audit_log"

    id = Column(String(64), primary_key=True)
    action = Column(String(100), nullable=False, index=True)
    node_id = Column(String(200), nullable=False, index=True)
    timestamp = Column(String(50), nullable=False)
    details = Column(Text, server_default="{}")


class SchemaVersionRecord(Base):
    """Persistent storage for schema registry entries (survives Railway redeploys)."""

    __tablename__ = "schema_versions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(200), unique=True, nullable=False, index=True)
    version = Column(BigInteger, nullable=False, server_default="1")
    schema_ = Column("schema", Text, nullable=False)
    hash = Column(String(64), nullable=False)
    created_at = Column(String(50), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
