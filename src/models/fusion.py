"""
Feature fusion module for combining temporal GRU output with GAT graph output.

Two fusion orders are compared in the ablation study:
  A. TemporalThenGraph (T→G): GRU processes sequence first, GAT propagates across graph.
     Rationale: Temporal evidence is accumulated first, then shared across prerequisites.
  B. GraphThenTemporal (G→T): GAT propagates initial features across graph, GRU processes
     the enriched features over time.
     Rationale: Pre-enriching features with prerequisite context before temporal modeling.

The plan specifies comparing both orders. Results are reported honestly from the ablation;
no ordering is assumed to be superior a priori.

Implementation:
  Both variants use the same GRU and GAT modules. The difference is in the
  data flow order. This is managed by the UMiKTGATModel (main model class).
"""

import torch
import torch.nn as nn
from torch import Tensor


class FusionLayer(nn.Module):
    """
    Fusion layer that combines GRU temporal output with GAT graph output.

    Concatenates the two representations along the feature dimension and
    projects to a shared hidden dimension.

    Args:
        temporal_dim: Dimension of GRU output.
        graph_dim: Dimension of GAT output.
        hidden_dim: Output dimension of fused representation.
        dropout_rate: Dropout on fusion output.
    """

    def __init__(
        self,
        temporal_dim: int = 64,
        graph_dim: int = 64,
        hidden_dim: int = 64,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(temporal_dim + graph_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, temporal_out: Tensor, graph_out: Tensor) -> Tensor:
        """
        Fuse temporal and graph representations.

        Args:
            temporal_out: GRU output. Shape: (n_concepts, temporal_dim)
            graph_out: GAT output. Shape: (n_concepts, graph_dim)

        Returns:
            Fused representation. Shape: (n_concepts, hidden_dim)
        """
        concat = torch.cat([temporal_out, graph_out], dim=-1)
        return self.proj(concat)
