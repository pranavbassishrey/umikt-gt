"""
Graph Attention Network (GAT) layer for prerequisite-aware learner-state propagation.

Implements two variants:
  1. GAT-Static: Standard learner-state-independent attention
       α_ij = softmax(LeakyReLU(W · concat(h_i, h_j)))
     This is the standard GAT formulation (Veličković et al. 2018).

  2. GAT-Dynamic: Learner-state-conditioned prerequisite attention (primary research point)
       α_ij,t = softmax(LeakyReLU(W · concat(h_i, h_j, s_i, s_j, e_ij)))
     Where:
       h_i, h_j = GRU hidden states (temporal evidence)
       s_i, s_j = current learner state vectors [M, MC, U, R] for concepts i,j
       e_ij = edge feature (dependency_weight from concept_graph.json)
     Justification: If s_i and s_j encode different mastery/misconception profiles,
     the influence of prerequisite j on concept i should differ from when both
     learners have the same profile. This is the core experimental hypothesis.
     Whether this actually improves performance is an empirical question answered
     by the ablation study, NOT assumed.

Implementation note: With only 7 concept nodes, this is a small dense graph.
We do NOT use PyTorch Geometric — the message passing is implemented directly
over the adjacency structure from concept_graph.json. This is sufficient for
this scale and removes a fragile optional dependency.

Both variants support multi-head attention with head-averaged output.
"""

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


N_CONCEPTS = 7


def load_adjacency_from_graph(graph_path: Path | None = None) -> tuple[list[tuple[int, int]], dict]:
    """
    Load edges and edge features from concept_graph.json.

    Returns:
        edges: List of (source_id, target_id) tuples.
        edge_features: Dict mapping (src, tgt) → dependency_weight float.
    """
    if graph_path is None:
        graph_path = (
            Path(__file__).parent.parent.parent / "data" / "concept_graph.json"
        )
    with open(graph_path) as f:
        g = json.load(f)

    edges = []
    edge_features = {}
    for e in g["edges"]:
        src, tgt = int(e["source"]), int(e["target"])
        edges.append((src, tgt))
        edge_features[(src, tgt)] = float(e["dependency_weight"])

    return edges, edge_features


class GraphAttentionLayer(nn.Module):
    """
    Single-head GAT layer operating over the concept prerequisite graph.

    Supports both GAT-Static and GAT-Dynamic variants.

    Args:
        node_dim: Dimension of each node's feature vector (GRU output dim).
        out_dim: Output dimension per head.
        learner_state_dim: Dimension of learner state vector per concept (4 for [M,MC,U,R]).
                           Only used in dynamic mode.
        edge_feature_dim: Dimension of edge features (1 for dependency_weight).
        dynamic: If True, use learner-state-conditioned attention (GAT-Dynamic).
        leaky_slope: Negative slope for LeakyReLU in attention.
        dropout_rate: Attention dropout.
        graph_path: Path to concept_graph.json.
    """

    def __init__(
        self,
        node_dim: int,
        out_dim: int,
        learner_state_dim: int = 4,
        edge_feature_dim: int = 1,
        dynamic: bool = True,
        leaky_slope: float = 0.2,
        dropout_rate: float = 0.2,
        graph_path: Path | None = None,
    ):
        super().__init__()

        self.node_dim = node_dim
        self.out_dim = out_dim
        self.dynamic = dynamic

        # Load graph structure (fixed, not learned)
        edges, edge_features = load_adjacency_from_graph(graph_path)
        self.edges = edges
        self.edge_feature_dict = edge_features

        # Precompute edge feature tensor (dependency weights)
        # Shape: (n_edges,)
        edge_weights = [edge_features.get(e, 0.5) for e in edges]
        self.register_buffer(
            "edge_weights",
            torch.tensor(edge_weights, dtype=torch.float32)
        )

        # Node feature projection
        self.W_node = nn.Linear(node_dim, out_dim, bias=False)

        # Attention computation
        # Static: concat(h_i, h_j) → scalar
        # Dynamic: concat(h_i, h_j, s_i, s_j, e_ij) → scalar
        if dynamic:
            attn_in_dim = 2 * out_dim + 2 * learner_state_dim + edge_feature_dim
        else:
            attn_in_dim = 2 * out_dim

        self.attn_fc = nn.Linear(attn_in_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_slope)
        self.dropout = nn.Dropout(p=dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_node.weight, gain=1.0 / math.sqrt(2))
        nn.init.xavier_uniform_(self.attn_fc.weight, gain=1.0 / math.sqrt(2))

    def forward(
        self,
        node_features: Tensor,
        learner_states: Tensor | None = None,
    ) -> Tensor:
        """
        Single forward pass of the GAT layer.

        Args:
            node_features: Concept hidden states from GRU.
                           Shape: (n_concepts, node_dim) = (7, node_dim)
            learner_states: Current learner state [M, MC, U, R] per concept.
                            Shape: (n_concepts, learner_state_dim) = (7, 4)
                            Required if dynamic=True.

        Returns:
            Updated node representations.
            Shape: (n_concepts, out_dim)
        """
        n = N_CONCEPTS

        # Project all node features
        h = self.W_node(node_features)  # (n, out_dim)

        # Initialize aggregated output as self-loop (residual)
        agg = h.clone()
        attn_count = torch.ones(n, device=node_features.device)  # for normalization

        # Collect attention logits for softmax normalization per target node
        # Group edges by target node
        edges_by_target: dict[int, list[int]] = {i: [] for i in range(n)}
        for edge_idx, (src, tgt) in enumerate(self.edges):
            edges_by_target[tgt].append(edge_idx)

        # Process each target node
        agg = torch.zeros(n, self.out_dim, device=node_features.device)
        # Add self-contributions (no attention — just pass through)
        agg = agg + h  # residual self-connection

        for tgt_id in range(n):
            src_edge_indices = edges_by_target[tgt_id]
            if not src_edge_indices:
                continue

            # Build attention inputs for all prerequisite edges → this target
            attn_inputs = []
            for edge_idx in src_edge_indices:
                src_id, _ = self.edges[edge_idx]
                ew = self.edge_weights[edge_idx:edge_idx + 1]  # (1,)

                if self.dynamic and learner_states is not None:
                    s_src = learner_states[src_id]  # (4,)
                    s_tgt = learner_states[tgt_id]  # (4,)
                    concat = torch.cat([h[src_id], h[tgt_id], s_src, s_tgt, ew])
                else:
                    concat = torch.cat([h[src_id], h[tgt_id]])

                attn_inputs.append(concat)

            # Stack: (n_src_edges, attn_in_dim)
            attn_stack = torch.stack(attn_inputs, dim=0)

            # Compute attention logits and scores
            logits = self.leaky_relu(self.attn_fc(attn_stack)).squeeze(-1)  # (n_src_edges,)
            scores = F.softmax(logits, dim=0)  # (n_src_edges,)
            scores = self.dropout(scores)

            # Weighted sum of source node representations
            for i, edge_idx in enumerate(src_edge_indices):
                src_id, _ = self.edges[edge_idx]
                agg[tgt_id] = agg[tgt_id] + scores[i] * h[src_id]

        return F.elu(agg)


class MultiHeadGAT(nn.Module):
    """
    Multi-head GAT over the concept prerequisite graph.

    Uses n_heads parallel attention heads, then averages their outputs.
    Averaging is used instead of concatenation to preserve output dimension
    consistency with the fusion layer, given the small graph size (7 nodes).

    Args:
        node_dim: Input node feature dimension (GRU output dim).
        out_dim: Output dimension per node (same as node_dim for residual compatibility).
        n_heads: Number of attention heads.
        learner_state_dim: Learner state vector dimension (4 for [M,MC,U,R]).
        dynamic: If True, use learner-state-conditioned attention (GAT-Dynamic).
        dropout_rate: Attention and output dropout.
        graph_path: Path to concept_graph.json.
    """

    def __init__(
        self,
        node_dim: int = 64,
        out_dim: int = 64,
        n_heads: int = 4,
        learner_state_dim: int = 4,
        dynamic: bool = True,
        dropout_rate: float = 0.2,
        graph_path: Path | None = None,
    ):
        super().__init__()

        self.n_heads = n_heads
        head_dim = max(1, out_dim // n_heads)

        self.heads = nn.ModuleList([
            GraphAttentionLayer(
                node_dim=node_dim,
                out_dim=head_dim,
                learner_state_dim=learner_state_dim,
                edge_feature_dim=1,
                dynamic=dynamic,
                dropout_rate=dropout_rate,
                graph_path=graph_path,
            )
            for _ in range(n_heads)
        ])

        # Project averaged head outputs back to out_dim
        actual_head_dim = head_dim
        self.out_proj = nn.Linear(actual_head_dim, out_dim, bias=True)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.output_dropout = nn.Dropout(p=dropout_rate)

    def forward(
        self,
        node_features: Tensor,
        learner_states: Tensor | None = None,
    ) -> Tensor:
        """
        Multi-head GAT forward pass.

        Args:
            node_features: Shape (n_concepts, node_dim)
            learner_states: Shape (n_concepts, 4) — optional for dynamic mode

        Returns:
            Updated node representations. Shape: (n_concepts, out_dim)
        """
        # Run all heads
        head_outputs = [head(node_features, learner_states) for head in self.heads]

        # Average across heads
        stacked = torch.stack(head_outputs, dim=0)  # (n_heads, n_concepts, head_dim)
        averaged = stacked.mean(dim=0)  # (n_concepts, head_dim)

        # Project and normalize
        out = self.out_proj(averaged)
        out = self.output_dropout(out)
        out = self.layer_norm(out + node_features[:, :out.shape[-1]])  # residual

        return out
