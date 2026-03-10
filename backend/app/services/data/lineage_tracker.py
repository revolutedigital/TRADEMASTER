"""End-to-end data lineage tracker.

Tracks the full lifecycle of data through the TradeMaster pipeline:
raw data -> features -> model training -> predictions -> trades.

Provides a DAG-based representation for dependency analysis, impact
assessment, compliance auditing, and future-data-leak detection.
"""

import json
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


class NodeType(str, Enum):
    RAW_DATA = "raw_data"
    FEATURE = "feature"
    MODEL = "model"
    PREDICTION = "prediction"
    TRADE = "trade"
    DATA_SOURCE = "data_source"
    TRANSFORM = "transform"


class EdgeType(str, Enum):
    DERIVED_FROM = "derived_from"
    TRAINED_ON = "trained_on"
    PREDICTED_BY = "predicted_by"
    TRIGGERED = "triggered"
    DEPENDS_ON = "depends_on"


@dataclass
class LineageNode:
    """A single artifact in the lineage graph."""

    id: str
    node_type: NodeType
    name: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)
    version: str = "1"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "version": self.version,
        }


@dataclass
class LineageEdge:
    """A directed dependency between two lineage nodes."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


class LineageDAG:
    """Directed acyclic graph representing data lineage relationships.

    Provides traversal, impact analysis, and visualization-ready export.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, LineageNode] = {}
        self._forward_edges: dict[str, list[LineageEdge]] = defaultdict(list)  # source -> edges
        self._reverse_edges: dict[str, list[LineageEdge]] = defaultdict(list)  # target -> edges

    def add_node(self, node: LineageNode) -> None:
        self._nodes[node.id] = node

    def add_edge(self, edge: LineageEdge) -> None:
        if edge.source_id not in self._nodes:
            raise ValueError(f"Source node '{edge.source_id}' not in DAG")
        if edge.target_id not in self._nodes:
            raise ValueError(f"Target node '{edge.target_id}' not in DAG")
        self._forward_edges[edge.source_id].append(edge)
        self._reverse_edges[edge.target_id].append(edge)

    def get_node(self, node_id: str) -> LineageNode | None:
        return self._nodes.get(node_id)

    def get_ancestors(self, node_id: str) -> list[LineageNode]:
        """Return all upstream nodes (transitive parents)."""
        visited: set[str] = set()
        result: list[LineageNode] = []
        queue = deque([node_id])
        while queue:
            current = queue.popleft()
            for edge in self._reverse_edges.get(current, []):
                if edge.source_id not in visited:
                    visited.add(edge.source_id)
                    result.append(self._nodes[edge.source_id])
                    queue.append(edge.source_id)
        return result

    def get_descendants(self, node_id: str) -> list[LineageNode]:
        """Return all downstream nodes (transitive children)."""
        visited: set[str] = set()
        result: list[LineageNode] = []
        queue = deque([node_id])
        while queue:
            current = queue.popleft()
            for edge in self._forward_edges.get(current, []):
                if edge.target_id not in visited:
                    visited.add(edge.target_id)
                    result.append(self._nodes[edge.target_id])
                    queue.append(edge.target_id)
        return result

    def get_roots(self) -> list[LineageNode]:
        """Return nodes with no incoming edges (original data sources)."""
        all_targets = {eid for edges in self._reverse_edges.values() for e in edges for eid in [e.target_id]}
        nodes_with_incoming = {e.target_id for edges in self._forward_edges.values() for e in edges}
        return [n for nid, n in self._nodes.items() if nid not in nodes_with_incoming]

    def to_visualization_data(self) -> dict:
        """Export graph in a format suitable for DAG visualization (e.g. D3.js).

        Returns:
            Dict with ``nodes`` and ``edges`` lists ready for JSON serialisation.
        """
        nodes = [
            {**node.to_dict(), "depth": self._compute_depth(node.id)}
            for node in self._nodes.values()
        ]
        edges = [
            edge.to_dict()
            for edge_list in self._forward_edges.values()
            for edge in edge_list
        ]
        return {"nodes": nodes, "edges": edges}

    def _compute_depth(self, node_id: str) -> int:
        """Compute the longest path from a root to this node."""
        if not self._reverse_edges.get(node_id):
            return 0
        return 1 + max(
            self._compute_depth(e.source_id)
            for e in self._reverse_edges[node_id]
        )

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return sum(len(el) for el in self._forward_edges.values())


class EndToEndLineageTracker:
    """Track data lineage across the full TradeMaster pipeline.

    Automatically records relationships as data flows from raw market
    feeds through feature engineering, model training, prediction
    generation, and trade execution.
    """

    def __init__(self) -> None:
        self._dag = LineageDAG()
        self._audit_log: list[dict] = []

    # ------------------------------------------------------------------
    # Node registration helpers
    # ------------------------------------------------------------------

    def register_data_source(self, source_id: str, name: str, **metadata: Any) -> LineageNode:
        """Register an external data source (exchange feed, API, etc.)."""
        node = LineageNode(
            id=source_id,
            node_type=NodeType.DATA_SOURCE,
            name=name,
            metadata=metadata,
        )
        self._dag.add_node(node)
        self._append_audit("register_data_source", node)
        logger.info("lineage_source_registered", source=source_id, name=name)
        return node

    def register_raw_data(
        self,
        data_id: str,
        name: str,
        source_id: str,
        timestamp: datetime | None = None,
        **metadata: Any,
    ) -> LineageNode:
        """Register an ingested raw data artifact and link it to its source."""
        node = LineageNode(
            id=data_id,
            node_type=NodeType.RAW_DATA,
            name=name,
            created_at=timestamp or datetime.now(timezone.utc),
            metadata=metadata,
        )
        self._dag.add_node(node)
        self._add_edge(source_id, data_id, EdgeType.DERIVED_FROM)
        self._append_audit("register_raw_data", node)
        return node

    def register_feature(
        self,
        feature_id: str,
        name: str,
        derived_from: list[str],
        version: str = "1",
        **metadata: Any,
    ) -> LineageNode:
        """Register a computed feature and its upstream dependencies."""
        node = LineageNode(
            id=feature_id,
            node_type=NodeType.FEATURE,
            name=name,
            version=version,
            metadata=metadata,
        )
        self._dag.add_node(node)
        for parent_id in derived_from:
            self._add_edge(parent_id, feature_id, EdgeType.DERIVED_FROM)
        self._append_audit("register_feature", node)
        return node

    def register_model(
        self,
        model_id: str,
        name: str,
        trained_on: list[str],
        version: str = "1",
        **metadata: Any,
    ) -> LineageNode:
        """Register a trained model and the features it was trained on."""
        node = LineageNode(
            id=model_id,
            node_type=NodeType.MODEL,
            name=name,
            version=version,
            metadata=metadata,
        )
        self._dag.add_node(node)
        for feature_id in trained_on:
            self._add_edge(feature_id, model_id, EdgeType.TRAINED_ON)
        self._append_audit("register_model", node)
        logger.info("lineage_model_registered", model=model_id, features=len(trained_on))
        return node

    def register_prediction(
        self,
        prediction_id: str,
        model_id: str,
        input_feature_ids: list[str],
        **metadata: Any,
    ) -> LineageNode:
        """Register a prediction and link it to the model and input features."""
        node = LineageNode(
            id=prediction_id,
            node_type=NodeType.PREDICTION,
            name=f"prediction:{prediction_id[:8]}",
            metadata=metadata,
        )
        self._dag.add_node(node)
        self._add_edge(model_id, prediction_id, EdgeType.PREDICTED_BY)
        for fid in input_feature_ids:
            self._add_edge(fid, prediction_id, EdgeType.DEPENDS_ON)
        self._append_audit("register_prediction", node)
        return node

    def register_trade(
        self,
        trade_id: str,
        prediction_id: str,
        **metadata: Any,
    ) -> LineageNode:
        """Register an executed trade and link it to the prediction that triggered it."""
        node = LineageNode(
            id=trade_id,
            node_type=NodeType.TRADE,
            name=f"trade:{trade_id[:8]}",
            metadata=metadata,
        )
        self._dag.add_node(node)
        self._add_edge(prediction_id, trade_id, EdgeType.TRIGGERED)
        self._append_audit("register_trade", node)
        return node

    # ------------------------------------------------------------------
    # Impact analysis
    # ------------------------------------------------------------------

    def impact_analysis(self, node_id: str) -> dict:
        """Determine what is affected if a given node fails or changes.

        Answers the question: "If this data source fails, which models /
        predictions / trades are affected?"

        Returns:
            Summary with lists of affected nodes grouped by type.
        """
        descendants = self._dag.get_descendants(node_id)
        source_node = self._dag.get_node(node_id)

        affected_by_type: dict[str, list[dict]] = defaultdict(list)
        for desc in descendants:
            affected_by_type[desc.node_type.value].append({
                "id": desc.id,
                "name": desc.name,
                "version": desc.version,
            })

        result = {
            "source_node": source_node.to_dict() if source_node else None,
            "total_affected": len(descendants),
            "affected_by_type": dict(affected_by_type),
        }
        logger.info(
            "impact_analysis_complete",
            node=node_id,
            total_affected=len(descendants),
        )
        return result

    # ------------------------------------------------------------------
    # Compliance: future data leak detection
    # ------------------------------------------------------------------

    def check_temporal_integrity(self, feature_id: str) -> dict:
        """Verify that a feature did not use data from the future.

        For a feature computed at time T, all ancestor raw data artifacts
        must have ``created_at <= T``.  Any violation indicates a potential
        look-ahead bias.

        Returns:
            Dict with ``is_valid``, ``violations`` list, and summary.
        """
        feature_node = self._dag.get_node(feature_id)
        if not feature_node:
            return {"is_valid": False, "error": f"Node '{feature_id}' not found"}

        feature_time = feature_node.created_at
        ancestors = self._dag.get_ancestors(feature_id)

        violations: list[dict] = []
        for ancestor in ancestors:
            if ancestor.node_type in (NodeType.RAW_DATA, NodeType.FEATURE):
                if ancestor.created_at > feature_time:
                    violations.append({
                        "ancestor_id": ancestor.id,
                        "ancestor_name": ancestor.name,
                        "ancestor_type": ancestor.node_type.value,
                        "ancestor_time": ancestor.created_at.isoformat(),
                        "feature_time": feature_time.isoformat(),
                        "leak_seconds": (ancestor.created_at - feature_time).total_seconds(),
                    })

        is_valid = len(violations) == 0
        result = {
            "feature_id": feature_id,
            "is_valid": is_valid,
            "violations": violations,
            "ancestors_checked": len(ancestors),
        }

        if not is_valid:
            logger.warning(
                "temporal_integrity_violation",
                feature=feature_id,
                violations=len(violations),
            )

        return result

    def validate_all_features(self) -> list[dict]:
        """Run temporal integrity checks across every feature node."""
        results: list[dict] = []
        for node_id, node in self._dag._nodes.items():
            if node.node_type == NodeType.FEATURE:
                results.append(self.check_temporal_integrity(node_id))
        return results

    # ------------------------------------------------------------------
    # Audit trail
    # ------------------------------------------------------------------

    def trace_to_source(self, node_id: str) -> list[dict]:
        """Trace a value back to its original raw data sources.

        Returns an ordered path from the queried node to each root
        ancestor, forming a complete audit trail.
        """
        node = self._dag.get_node(node_id)
        if not node:
            return []

        ancestors = self._dag.get_ancestors(node_id)
        trail = [node.to_dict()] + [a.to_dict() for a in ancestors]

        logger.debug("trace_to_source", node=node_id, depth=len(trail))
        return trail

    def get_audit_log(self, limit: int = 100) -> list[dict]:
        """Return the most recent audit log entries."""
        return self._audit_log[-limit:]

    # ------------------------------------------------------------------
    # DAG visualisation & stats
    # ------------------------------------------------------------------

    def get_visualization_data(self) -> dict:
        """Export the full lineage DAG for front-end rendering."""
        return self._dag.to_visualization_data()

    def get_subgraph(self, node_id: str) -> dict:
        """Export the sub-DAG rooted at ``node_id`` (ancestors + descendants)."""
        node = self._dag.get_node(node_id)
        if not node:
            return {"nodes": [], "edges": []}

        related_ids = {node_id}
        for n in self._dag.get_ancestors(node_id):
            related_ids.add(n.id)
        for n in self._dag.get_descendants(node_id):
            related_ids.add(n.id)

        nodes = [
            self._dag._nodes[nid].to_dict()
            for nid in related_ids
            if nid in self._dag._nodes
        ]
        edges = [
            e.to_dict()
            for edge_list in self._dag._forward_edges.values()
            for e in edge_list
            if e.source_id in related_ids and e.target_id in related_ids
        ]
        return {"nodes": nodes, "edges": edges}

    def get_stats(self) -> dict:
        """Return summary statistics about the lineage graph."""
        type_counts: dict[str, int] = defaultdict(int)
        for node in self._dag._nodes.values():
            type_counts[node.node_type.value] += 1

        return {
            "total_nodes": self._dag.node_count,
            "total_edges": self._dag.edge_count,
            "nodes_by_type": dict(type_counts),
            "audit_log_size": len(self._audit_log),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def persist(self) -> None:
        """Save the full lineage graph to PostgreSQL for durable storage."""
        try:
            from sqlalchemy import text

            from app.models.base import async_session

            async with async_session() as session:
                # Persist nodes
                for node in self._dag._nodes.values():
                    await session.execute(
                        text("""
                            INSERT INTO lineage_nodes
                                (id, node_type, name, created_at, version, metadata)
                            VALUES
                                (:id, :node_type, :name, :created_at, :version, :metadata)
                            ON CONFLICT (id) DO UPDATE SET
                                metadata = EXCLUDED.metadata,
                                version = EXCLUDED.version
                        """),
                        {
                            "id": node.id,
                            "node_type": node.node_type.value,
                            "name": node.name,
                            "created_at": node.created_at,
                            "version": node.version,
                            "metadata": json.dumps(node.metadata),
                        },
                    )

                # Persist edges
                for edge_list in self._dag._forward_edges.values():
                    for edge in edge_list:
                        await session.execute(
                            text("""
                                INSERT INTO lineage_edges
                                    (source_id, target_id, edge_type, created_at, metadata)
                                VALUES
                                    (:source_id, :target_id, :edge_type, :created_at, :metadata)
                                ON CONFLICT (source_id, target_id, edge_type) DO NOTHING
                            """),
                            {
                                "source_id": edge.source_id,
                                "target_id": edge.target_id,
                                "edge_type": edge.edge_type.value,
                                "created_at": edge.created_at,
                                "metadata": json.dumps(edge.metadata),
                            },
                        )

                # Persist audit log
                for entry in self._audit_log:
                    await session.execute(
                        text("""
                            INSERT INTO lineage_audit_log
                                (id, action, node_id, timestamp, details)
                            VALUES
                                (:id, :action, :node_id, :timestamp, :details)
                            ON CONFLICT (id) DO NOTHING
                        """),
                        {
                            "id": entry["id"],
                            "action": entry["action"],
                            "node_id": entry["node_id"],
                            "timestamp": entry["timestamp"],
                            "details": json.dumps(entry),
                        },
                    )

                await session.commit()
                logger.info(
                    "lineage_persisted",
                    nodes=self._dag.node_count,
                    edges=self._dag.edge_count,
                )
        except Exception as exc:
            logger.warning("lineage_persist_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_edge(self, source_id: str, target_id: str, edge_type: EdgeType, **metadata: Any) -> None:
        """Safely add an edge, skipping if either node is missing."""
        if self._dag.get_node(source_id) and self._dag.get_node(target_id):
            edge = LineageEdge(
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                metadata=metadata,
            )
            self._dag.add_edge(edge)
        else:
            logger.debug(
                "lineage_edge_skipped",
                source=source_id,
                target=target_id,
                reason="node_not_found",
            )

    def _append_audit(self, action: str, node: LineageNode) -> None:
        self._audit_log.append({
            "id": uuid.uuid4().hex,
            "action": action,
            "node_id": node.id,
            "node_type": node.node_type.value,
            "node_name": node.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


lineage_tracker = EndToEndLineageTracker()
