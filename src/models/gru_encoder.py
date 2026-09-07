"""
GRU temporal encoder for the UMiKT-GAT model.

Implements H_t = GRU(X_t, H_{t-1}).

Design decisions (documented here, not presented as novel):
  - GRU chosen over LSTM: fewer parameters (2/3 gates vs 3/4), similar
    empirical performance on sequential KT tasks (Chung et al. 2014).
  - GRU chosen over Transformers: dataset size (~6k-9k rows) does not
    justify quadratic attention costs; GRU is standard for this scale.
  - MC-Dropout implemented via training-mode inference (Gal & Ghahramani 2016).
    Dropout is applied between GRU layers, not within GRU cells (per PyTorch default).
    This is a known limitation — full variational dropout within cells is more
    principled but harder to implement in standard PyTorch.

The module supports:
  - Single-sequence inference (one learner at a time)
  - Batched inference (padded sequences)
  - MC-Dropout forward passes for uncertainty estimation
"""

import torch
import torch.nn as nn
from torch import Tensor


class GRUEncoder(nn.Module):
    """
    Multi-layer GRU encoder with MC-Dropout support.

    Args:
        input_dim: Dimension of input feature vector (must match feature_builder.INPUT_DIM).
        hidden_dim: GRU hidden state dimension.
        num_layers: Number of stacked GRU layers.
        dropout_rate: Dropout probability applied between GRU layers.
                      Also used for MC-Dropout uncertainty estimation.
        bidirectional: If True, use a bidirectional GRU. Default False.
                       Not recommended for causal KT (would use future information).
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Additional dropout after the final GRU output
        # This is the layer used for MC-Dropout inference
        self.output_dropout = nn.Dropout(p=dropout_rate)

        # Layer norm on output for training stability
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.layer_norm = nn.LayerNorm(out_dim)

        self.output_dim = out_dim

    def forward(
        self,
        x: Tensor,
        h_prev: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass through the GRU encoder.

        Args:
            x: Input tensor. Shape: (batch, seq_len, input_dim) for batched,
               or (seq_len, input_dim) for unbatched (will be unsqueezed).
            h_prev: Previous hidden state. Shape: (num_layers * directions, batch, hidden_dim).
                    If None, initialized to zeros.

        Returns:
            Tuple of:
              - output: Hidden states for all time steps.
                        Shape: (batch, seq_len, hidden_dim * directions)
              - h_n: Final hidden state. Shape: (num_layers * directions, batch, hidden_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, seq_len, input_dim)

        output, h_n = self.gru(x, h_prev)

        # Apply dropout + layer norm to output
        output = self.output_dropout(output)
        output = self.layer_norm(output)

        return output, h_n

    def init_hidden(self, batch_size: int = 1, device: torch.device | None = None) -> Tensor:
        """Initialize hidden state to zeros."""
        if device is None:
            device = next(self.parameters()).device
        directions = 2 if self.bidirectional else 1
        return torch.zeros(
            self.num_layers * directions, batch_size, self.hidden_dim,
            device=device
        )
