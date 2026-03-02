"""Graph-based relationship store for crypto assets.

Models relationships between assets, traders, and strategies
for correlation analysis and influence propagation.
Uses an in-memory graph (production would use Neo4j).
"""

from dataclasses import dataclass, field
from enum import Enum

from app.core.logging import get_logger

logger = get_logger(__name__)


class RelationshipType(str, Enum):
    CORRELATED = "correlated"
    INVERSE_CORRELATED = "inverse_correlated"
    FOLLOWS = "follows"
    TRADES = "trades"
    INFLUENCES = "influences"


@dataclass
class Node:
    id: str
    type: str  # "asset", "trader", "strategy"
    properties: dict = field(default_factory=dict)


@dataclass
class Edge:
    source: str
    target: str
    relationship: RelationshipType
    weight: float = 1.0
    properties: dict = field(default_factory=dict)


class GraphStore:
    """In-memory graph store for relationship modeling."""

    def __init__(self):
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []

    def add_node(self, node_id: str, node_type: str, **properties) -> Node:
        node = Node(id=node_id, type=node_type, properties=properties)
        self._nodes[node_id] = node
        return node

    def add_edge(self, source: str, target: str, relationship: RelationshipType, weight: float = 1.0, **properties) -> Edge:
        edge = Edge(source=source, target=target, relationship=relationship, weight=weight, properties=properties)
        self._edges.append(edge)
        return edge

    def get_neighbors(self, node_id: str, relationship: RelationshipType | None = None) -> list[tuple[Node, Edge]]:
        """Get all neighbors of a node."""
        neighbors = []
        for edge in self._edges:
            if relationship and edge.relationship != relationship:
                continue
            if edge.source == node_id and edge.target in self._nodes:
                neighbors.append((self._nodes[edge.target], edge))
            elif edge.target == node_id and edge.source in self._nodes:
                neighbors.append((self._nodes[edge.source], edge))
        return neighbors

    def find_path(self, source: str, target: str, max_depth: int = 5) -> list[str] | None:
        """BFS to find shortest path between nodes."""
        if source not in self._nodes or target not in self._nodes:
            return None
        
        visited = set()
        queue = [(source, [source])]
        
        while queue:
            current, path = queue.pop(0)
            if current == target:
                return path
            if current in visited or len(path) > max_depth:
                continue
            visited.add(current)
            
            for neighbor, _ in self.get_neighbors(current):
                if neighbor.id not in visited:
                    queue.append((neighbor.id, path + [neighbor.id]))
        
        return None

    def get_correlation_cluster(self, asset_id: str) -> list[dict]:
        """Get cluster of correlated assets."""
        neighbors = self.get_neighbors(asset_id, RelationshipType.CORRELATED)
        return [
            {
                "asset": n.id,
                "correlation": e.weight,
                "properties": n.properties,
            }
            for n, e in sorted(neighbors, key=lambda x: x[1].weight, reverse=True)
        ]

    def get_influence_score(self, node_id: str) -> float:
        """Calculate influence score (simplified PageRank)."""
        incoming = sum(1 for e in self._edges if e.target == node_id)
        outgoing = sum(1 for e in self._edges if e.source == node_id)
        total_edges = len(self._edges) or 1
        return (incoming + 0.5 * outgoing) / total_edges

    def get_stats(self) -> dict:
        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "node_types": list(set(n.type for n in self._nodes.values())),
            "relationship_types": list(set(e.relationship.value for e in self._edges)),
        }

    def initialize_crypto_graph(self):
        """Initialize with known crypto asset relationships."""
        # Add assets
        self.add_node("BTC", "asset", name="Bitcoin", market_cap_rank=1)
        self.add_node("ETH", "asset", name="Ethereum", market_cap_rank=2)
        self.add_node("BNB", "asset", name="Binance Coin", market_cap_rank=4)
        self.add_node("SOL", "asset", name="Solana", market_cap_rank=5)
        self.add_node("USDT", "asset", name="Tether", is_stablecoin=True)

        # Add correlations
        self.add_edge("BTC", "ETH", RelationshipType.CORRELATED, weight=0.85)
        self.add_edge("BTC", "BNB", RelationshipType.CORRELATED, weight=0.75)
        self.add_edge("BTC", "SOL", RelationshipType.CORRELATED, weight=0.70)
        self.add_edge("ETH", "SOL", RelationshipType.CORRELATED, weight=0.80)

        # Add influences
        self.add_edge("BTC", "ETH", RelationshipType.INFLUENCES, weight=0.90)
        self.add_edge("BTC", "BNB", RelationshipType.INFLUENCES, weight=0.70)
        self.add_edge("BTC", "SOL", RelationshipType.INFLUENCES, weight=0.65)

        logger.info("crypto_graph_initialized", nodes=len(self._nodes), edges=len(self._edges))


graph_store = GraphStore()
