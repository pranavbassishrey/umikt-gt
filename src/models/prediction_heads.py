"""
Multi-task prediction heads for the UMiKT-GAT model.

Implements four output heads from a shared representation:
  1. MasteryHead: Predict mastery M_i ∈ [0,1] for each concept
  2. MisconceptionHead: Predict misconception class (7 classes)
  3. RetentionHead: Predict retention/forgetting risk R_i ∈ [0,1]
  4. (Uncertainty is estimated externally via MC-Dropout, not a separate head)

Design decisions:
  - Mastery: Sigmoid output, BCE loss. Target is continuous [0,1] ground truth.
    This formulation is standard in Knowledge Tracing (e.g., DKT).
  - Misconception: 7-class softmax, cross-entropy loss with class weighting.
    Single-label per interaction (per taxonomy design doc).
    Class weights address class imbalance (majority is 'none').
  - Retention: Sigmoid output, MSE loss. Target is the ground-truth decay value.
    This is the rule-based exponential decay output, not a learned prediction
    of future performance. The distinction is documented in the DATASET_CARD.

All heads use a small 2-layer MLP to avoid over-fitting on the small synthetic dataset.
"""

import torch
import torch.nn as nn
from torch import Tensor

N_MISCONCEPTION_CLASSES = 7


class MasteryHead(nn.Module):
    """
    Mastery prediction head: M_i ∈ [0,1].

    Architecture: Linear(hidden_dim → hidden_dim//2) → ReLU → Linear → Sigmoid
    Loss: Binary Cross-Entropy (BCELoss) treating mastery as a probability.
    """

    def __init__(self, hidden_dim: int = 64, dropout_rate: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, h: Tensor) -> Tensor:
        """
        Args:
            h: Shared representation. Shape: (..., hidden_dim)
        Returns:
            Mastery prediction. Shape: (..., 1)
        """
        return self.net(h)


class MisconceptionHead(nn.Module):
    """
    Misconception classification head: 7-class softmax.

    Architecture: Linear(hidden_dim → hidden_dim//2) → ReLU → Linear(7)
    Loss: CrossEntropyLoss with class weights (class 0 'none' is majority).

    The class weights are computed from training data class frequencies.
    Default weights below are priors — they should be recomputed on actual training
    data and passed in via set_class_weights().
    """

    def __init__(self, hidden_dim: int = 64, dropout_rate: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim // 2, N_MISCONCEPTION_CLASSES),
        )
        # Default class weights (will be overridden by actual training distribution)
        # Class 0 (none) is majority → lower weight; minority classes get higher weight
        self.register_buffer(
            "class_weights",
            torch.tensor([0.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=torch.float32)
        )

    def set_class_weights(self, weights: Tensor):
        """Update class weights from actual training data distribution."""
        self.class_weights = weights.to(self.class_weights.device)

    def forward(self, h: Tensor) -> Tensor:
        """
        Args:
            h: Shared representation. Shape: (..., hidden_dim)
        Returns:
            Class logits. Shape: (..., n_classes)
            Note: logits, not probabilities. CrossEntropyLoss expects logits.
        """
        return self.net(h)

    def get_loss_fn(self) -> nn.Module:
        """Return the loss function configured with class weights."""
        return nn.CrossEntropyLoss(weight=self.class_weights)


class RetentionHead(nn.Module):
    """
    Retention/forgetting prediction head: R_i ∈ [0,1].

    Predicts the retention state, which combines the ground-truth exponential
    decay value in training. The head learns to predict decay risk from
    the learned representation.

    Loss: MSE (targets are continuous [0,1] decay values).
    """

    def __init__(self, hidden_dim: int = 64, dropout_rate: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, h: Tensor) -> Tensor:
        """
        Args:
            h: Shared representation. Shape: (..., hidden_dim)
        Returns:
            Retention prediction. Shape: (..., 1)
        """
        return self.net(h)
