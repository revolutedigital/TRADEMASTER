"""Graph Neural Network for modeling cryptocurrency asset relationships."""

import math
from dataclasses import dataclass

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GNNConfig:
    """Configuration for Graph Neural Network."""
    node_feature_dim: int = 32
    hidden_dim: int = 64
    output_dim: int = 16
    num_layers: int = 3
    dropout: float = 0.1
    aggregation: str = "mean"  # mean, sum, max
    use_edge_features: bool = True
    edge_feature_dim: int = 4


class GraphData:
    """Represents a graph of cryptocurrency assets."""

    def __init__(self):
        self.nodes: dict[str, np.ndarray] = {}
        self.edges: list[tuple[str, str, np.ndarray]] = []
        self.adjacency: dict[str, list[str]] = {}

    def add_node(self, node_id: str, features: np.ndarray) -> None:
        self.nodes[node_id] = features
        if node_id not in self.adjacency:
            self.adjacency[node_id] = []

    def add_edge(self, src: str, dst: str, features: np.ndarray | None = None) -> None:
        if features is None:
            features = np.ones(4)
        self.edges.append((src, dst, features))
        if src not in self.adjacency:
            self.adjacency[src] = []
        if dst not in self.adjacency:
            self.adjacency[dst] = []
        self.adjacency[src].append(dst)
        self.adjacency[dst].append(src)  # Undirected

    def get_neighbors(self, node_id: str) -> list[str]:
        return self.adjacency.get(node_id, [])

    def get_edge_features(self, src: str, dst: str) -> np.ndarray | None:
        for s, d, f in self.edges:
            if (s == src and d == dst) or (s == dst and d == src):
                return f
        return None

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)


class GNNLayer:
    """Single Graph Neural Network layer with message passing."""

    def __init__(self, input_dim: int, output_dim: int, edge_dim: int = 4,
                 use_edge_features: bool = True, aggregation: str = "mean"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_edge_features = use_edge_features
        self.aggregation = aggregation

        # Message network weights
        msg_input = input_dim * 2 + (edge_dim if use_edge_features else 0)
        limit = math.sqrt(6.0 / (msg_input + output_dim))
        self._msg_w = np.random.uniform(-limit, limit, (msg_input, output_dim))
        self._msg_b = np.zeros(output_dim)

        # Update network weights
        limit = math.sqrt(6.0 / (input_dim + output_dim + output_dim))
        self._upd_w = np.random.uniform(-limit, limit, (input_dim + output_dim, output_dim))
        self._upd_b = np.zeros(output_dim)

    def forward(self, graph: GraphData) -> dict[str, np.ndarray]:
        """Message passing: message -> aggregate -> update."""
        new_features = {}

        for node_id, node_feat in graph.nodes.items():
            neighbors = graph.get_neighbors(node_id)

            if not neighbors:
                # Self-loop for isolated nodes
                new_features[node_id] = np.tanh(
                    np.concatenate([node_feat, np.zeros(self.output_dim)]) @ self._upd_w + self._upd_b
                )
                continue

            # Compute messages from neighbors
            messages = []
            for neighbor_id in neighbors:
                neighbor_feat = graph.nodes[neighbor_id]
                msg_input = [node_feat, neighbor_feat]

                if self.use_edge_features:
                    edge_feat = graph.get_edge_features(node_id, neighbor_id)
                    if edge_feat is not None:
                        msg_input.append(edge_feat)
                    else:
                        msg_input.append(np.zeros(4))

                msg_concat = np.concatenate(msg_input)
                # Pad or truncate to match weight dimensions
                if len(msg_concat) < self._msg_w.shape[0]:
                    msg_concat = np.pad(msg_concat, (0, self._msg_w.shape[0] - len(msg_concat)))
                elif len(msg_concat) > self._msg_w.shape[0]:
                    msg_concat = msg_concat[:self._msg_w.shape[0]]

                message = np.tanh(msg_concat @ self._msg_w + self._msg_b)
                messages.append(message)

            # Aggregate messages
            messages_array = np.stack(messages)
            if self.aggregation == "mean":
                aggregated = np.mean(messages_array, axis=0)
            elif self.aggregation == "sum":
                aggregated = np.sum(messages_array, axis=0)
            elif self.aggregation == "max":
                aggregated = np.max(messages_array, axis=0)
            else:
                aggregated = np.mean(messages_array, axis=0)

            # Update node features
            update_input = np.concatenate([node_feat, aggregated])
            if len(update_input) < self._upd_w.shape[0]:
                update_input = np.pad(update_input, (0, self._upd_w.shape[0] - len(update_input)))
            elif len(update_input) > self._upd_w.shape[0]:
                update_input = update_input[:self._upd_w.shape[0]]

            new_features[node_id] = np.tanh(update_input @ self._upd_w + self._upd_b)

        return new_features


class CryptoGraphNN:
    """
    Graph Neural Network for cryptocurrency market analysis.

    Models relationships between crypto assets using:
    - Price correlation edges
    - Volume flow edges
    - Sector/category edges
    - On-chain transaction edges

    Outputs:
    - Node embeddings for asset similarity
    - Graph-level predictions for market state
    - Edge predictions for correlation forecasting
    """

    def __init__(self, config: GNNConfig | None = None):
        self.config = config or GNNConfig()

        self.layers: list[GNNLayer] = []
        dims = [self.config.node_feature_dim] + [self.config.hidden_dim] * (self.config.num_layers - 1) + [self.config.output_dim]

        for i in range(len(dims) - 1):
            self.layers.append(GNNLayer(
                input_dim=dims[i],
                output_dim=dims[i + 1],
                edge_dim=self.config.edge_feature_dim,
                use_edge_features=self.config.use_edge_features,
                aggregation=self.config.aggregation,
            ))

        # Readout layer for graph-level prediction
        limit = math.sqrt(6.0 / (self.config.output_dim + 1))
        self._readout_w = np.random.uniform(-limit, limit, (self.config.output_dim, 3))  # bullish, neutral, bearish
        self._readout_b = np.zeros(3)

        logger.info("gnn_initialized", layers=len(self.layers), config=str(self.config))

    def forward(self, graph: GraphData) -> dict:
        """Forward pass through all GNN layers."""
        current_graph = graph

        # Multi-layer message passing
        layer_embeddings = []
        for i, layer in enumerate(self.layers):
            new_features = layer.forward(current_graph)

            # Update graph with new features
            updated_graph = GraphData()
            for node_id in current_graph.nodes:
                feat = new_features.get(node_id, current_graph.nodes[node_id])
                # Ensure correct dimension
                if len(feat) < self.layers[min(i + 1, len(self.layers) - 1)].input_dim:
                    feat = np.pad(feat, (0, self.layers[min(i + 1, len(self.layers) - 1)].input_dim - len(feat)))
                elif len(feat) > self.layers[min(i + 1, len(self.layers) - 1)].input_dim:
                    feat = feat[:self.layers[min(i + 1, len(self.layers) - 1)].input_dim]
                updated_graph.add_node(node_id, feat)

            for src, dst, edge_feat in current_graph.edges:
                updated_graph.add_edge(src, dst, edge_feat)

            current_graph = updated_graph
            layer_embeddings.append({k: v.copy() for k, v in new_features.items()})

        # Final node embeddings
        node_embeddings = {node_id: current_graph.nodes[node_id] for node_id in current_graph.nodes}

        # Graph-level readout (mean pooling)
        if node_embeddings:
            all_embeddings = np.stack(list(node_embeddings.values()))
            graph_embedding = np.mean(all_embeddings, axis=0)
        else:
            graph_embedding = np.zeros(self.config.output_dim)

        # Ensure correct dimension for readout
        if len(graph_embedding) < self._readout_w.shape[0]:
            graph_embedding = np.pad(graph_embedding, (0, self._readout_w.shape[0] - len(graph_embedding)))
        elif len(graph_embedding) > self._readout_w.shape[0]:
            graph_embedding = graph_embedding[:self._readout_w.shape[0]]

        # Market state prediction
        logits = graph_embedding @ self._readout_w + self._readout_b
        exp_logits = np.exp(logits - np.max(logits))
        market_probs = exp_logits / (np.sum(exp_logits) + 1e-8)

        return {
            "node_embeddings": {k: v.tolist() for k, v in node_embeddings.items()},
            "graph_embedding": graph_embedding.tolist(),
            "market_state": {
                "bullish": float(market_probs[0]),
                "neutral": float(market_probs[1]),
                "bearish": float(market_probs[2]),
            },
            "prediction": ["bullish", "neutral", "bearish"][int(np.argmax(market_probs))],
            "confidence": float(np.max(market_probs)),
        }

    def build_crypto_graph(self, price_data: dict[str, np.ndarray],
                           correlations: dict[tuple[str, str], float] | None = None) -> GraphData:
        """Build a graph from cryptocurrency market data."""
        graph = GraphData()

        # Add nodes (each crypto asset)
        for symbol, prices in price_data.items():
            if len(prices) == 0:
                continue

            # Feature engineering per node
            returns = np.diff(prices) / prices[:-1] if len(prices) > 1 else np.zeros(1)
            features = np.array([
                float(np.mean(returns)),           # avg return
                float(np.std(returns)),             # volatility
                float(np.mean(prices[-5:])),        # recent price mean
                float(prices[-1] / prices[0] - 1) if prices[0] != 0 else 0,  # total return
                float(np.max(prices) / np.min(prices) - 1) if np.min(prices) != 0 else 0,  # range
            ])

            # Pad/truncate to node_feature_dim
            if len(features) < self.config.node_feature_dim:
                features = np.pad(features, (0, self.config.node_feature_dim - len(features)))
            else:
                features = features[:self.config.node_feature_dim]

            graph.add_node(symbol, features)

        # Add edges based on correlations
        if correlations:
            for (src, dst), corr in correlations.items():
                if src in graph.nodes and dst in graph.nodes:
                    edge_features = np.array([
                        abs(corr),     # correlation strength
                        1.0 if corr > 0 else 0.0,  # positive correlation flag
                        corr,          # raw correlation
                        1.0,           # edge weight
                    ])
                    if abs(corr) > 0.3:  # Only add significant correlations
                        graph.add_edge(src, dst, edge_features)
        else:
            # Auto-compute correlations
            symbols = list(price_data.keys())
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    prices_i = price_data[symbols[i]]
                    prices_j = price_data[symbols[j]]
                    min_len = min(len(prices_i), len(prices_j))
                    if min_len > 10:
                        corr = float(np.corrcoef(prices_i[-min_len:], prices_j[-min_len:])[0, 1])
                        if abs(corr) > 0.3:
                            edge_features = np.array([abs(corr), 1.0 if corr > 0 else 0.0, corr, 1.0])
                            graph.add_edge(symbols[i], symbols[j], edge_features)

        logger.info("crypto_graph_built", nodes=graph.num_nodes, edges=graph.num_edges)
        return graph

    def find_similar_assets(self, graph: GraphData, target_symbol: str, top_k: int = 5) -> list[dict]:
        """Find assets most similar to target based on GNN embeddings."""
        result = self.forward(graph)
        embeddings = result["node_embeddings"]

        if target_symbol not in embeddings:
            return []

        target_emb = np.array(embeddings[target_symbol])
        similarities = []

        for symbol, emb in embeddings.items():
            if symbol == target_symbol:
                continue
            emb = np.array(emb)
            # Cosine similarity
            dot = np.dot(target_emb, emb)
            norm = np.linalg.norm(target_emb) * np.linalg.norm(emb)
            sim = float(dot / (norm + 1e-8))
            similarities.append({"symbol": symbol, "similarity": sim})

        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]

    def detect_clusters(self, graph: GraphData, threshold: float = 0.7) -> list[list[str]]:
        """Detect asset clusters based on embedding similarity."""
        result = self.forward(graph)
        embeddings = result["node_embeddings"]

        symbols = list(embeddings.keys())
        n = len(symbols)
        if n == 0:
            return []

        # Build similarity matrix
        emb_matrix = np.array([embeddings[s] for s in symbols])
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8
        normalized = emb_matrix / norms
        sim_matrix = normalized @ normalized.T

        # Simple clustering via connected components on similarity graph
        visited = set()
        clusters = []

        for i in range(n):
            if i in visited:
                continue
            cluster = [symbols[i]]
            visited.add(i)
            queue = [i]

            while queue:
                current = queue.pop(0)
                for j in range(n):
                    if j not in visited and sim_matrix[current, j] > threshold:
                        cluster.append(symbols[j])
                        visited.add(j)
                        queue.append(j)

            clusters.append(cluster)

        logger.info("clusters_detected", n_clusters=len(clusters),
                    sizes=[len(c) for c in clusters])
        return clusters


# Module-level instance
crypto_gnn = CryptoGraphNN()
