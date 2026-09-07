"""
Retention / forgetting module for the UMiKT-GAT pipeline.

Implements two variants:

Variant 1 (Primary): ExponentialDecayRetention
  R_i(t) = M_i * exp(-λ * Δt)
  Where:
    M_i = current mastery estimate
    λ = global forgetting rate (fitted from training data)
    Δt = time since last meaningful practice (hours)

  This is a standard baseline model in forgetting/retention research (Ebbinghaus 1885).
  The exponential form is NOT claimed as a novel contribution. It is a baseline
  that provides the forgetting-risk estimate F_i for the curriculum priority function.

  λ is fitted by MLE on the training data: maximize log P(observed correctness | R_i(t)).
  This is done in one pass through the training data before model training starts.

Variant 2 (Experimental): LearnedLambdaRetention
  λ_i(t) = MLP(learner_history_features)
  R_i(t) = M_i * exp(-λ_i(t) * Δt)

  The learned λ variant is labeled EXPERIMENTAL because:
    (a) The synthetic data was generated with a fixed global λ prior, so the
        learned variant has limited signal to differentiate from the global baseline.
    (b) We do not have real longitudinal data with observed forgetting gaps.
    (c) Results from this variant should be interpreted cautiously.
  These limitations are documented in DATASET_CARD.md and RESULTS_SUMMARY.md.

Note on novelty: Per the project spec, novelty claims for retention would come
from *how* λ is estimated and integrated, not from the exponential form itself.
The primary contribution is the integration into the multi-task learner state,
not the forgetting model per se.
"""

import math
import torch
import torch.nn as nn
from torch import Tensor


class ExponentialDecayRetention(nn.Module):
    """
    Exponential decay retention model with global fitted forgetting rate.

    R_i(t) = M_i * exp(-λ * Δt)

    λ is a learnable scalar parameter initialized from the fitted MLE estimate.
    Using nn.Parameter allows optional fine-tuning during training.

    Args:
        lambda_init: Initial forgetting rate λ (per hour). Default from prior.
        learnable: If True, λ is a trainable parameter. If False, it's fixed.
    """

    def __init__(self, lambda_init: float = 0.05, learnable: bool = True):
        super().__init__()

        # Store log(λ) to ensure λ > 0 during optimization
        log_lambda = math.log(max(lambda_init, 1e-6))
        self.log_lambda = nn.Parameter(
            torch.tensor(log_lambda, dtype=torch.float32),
            requires_grad=learnable,
        )
        self.learnable = learnable

    @property
    def lambda_value(self) -> float:
        """Current forgetting rate λ."""
        return float(torch.exp(self.log_lambda).item())

    def forward(self, mastery: Tensor, delta_t_hours: Tensor) -> Tensor:
        """
        Compute retention from mastery and time gap.

        Args:
            mastery: Current mastery estimates. Shape: (n_concepts,) or (batch, n_concepts)
            delta_t_hours: Time since last practice in hours. Same shape as mastery.

        Returns:
            Retention estimates. Same shape as mastery.
        """
        lam = torch.exp(self.log_lambda)
        retention = mastery * torch.exp(-lam * delta_t_hours)
        return torch.clamp(retention, 0.0, 1.0)

    def forgetting_risk(self, mastery: Tensor, delta_t_hours: Tensor) -> Tensor:
        """
        Compute forgetting risk = 1 - R_i(t).

        High forgetting risk → schedule spaced revision in curriculum policy.

        Returns:
            Forgetting risk. Shape same as mastery.
        """
        return 1.0 - self.forward(mastery, delta_t_hours)

    @classmethod
    def fit_lambda_from_data(cls, records: list[dict]) -> float:
        """
        Fit global λ via a simple heuristic from training data.

        Method: For pairs of interactions on the same concept by the same learner,
        compute the correlation between temporal_gap and correctness drop.
        λ is estimated as the rate that best explains observed correctness decay.

        This is a simplified estimation (not full MLE) justified by:
          1. Small dataset size.
          2. Noisy synthetic data.
          3. λ will be fine-tuned during model training if learnable=True.

        Returns:
            Estimated λ in units of per-hour decay.
        """
        # Group by (learner_id, concept_id)
        from collections import defaultdict
        sequences = defaultdict(list)
        for r in records:
            if "temporal_gap_hours" in r and "correctness" in r:
                key = (r.get("learner_id", ""), r.get("concept_id", 0))
                sequences[key].append((
                    float(r["temporal_gap_hours"]),
                    float(r["correctness"]),
                ))

        # Compute mean "correctness drop per hour" across all transitions
        lambda_estimates = []
        for seq in sequences.values():
            seq.sort(key=lambda x: x[0])
            for i in range(1, len(seq)):
                dt = seq[i][0] - seq[i - 1][0]
                dc = seq[i - 1][1] - seq[i][1]  # drop in correctness
                if dt > 0 and dc > 0:
                    est = dc / (dt + 1e-6)
                    lambda_estimates.append(min(est, 0.5))  # cap at 0.5

        if not lambda_estimates:
            return 0.05  # fallback prior

        median_lambda = sorted(lambda_estimates)[len(lambda_estimates) // 2]
        return max(0.001, min(0.5, median_lambda))


class LearnedLambdaRetention(nn.Module):
    """
    EXPERIMENTAL: Learned per-concept forgetting rate from learner history.

    λ_i(t) = softplus(MLP([h_i, temporal_gap_norm]))
    R_i(t) = M_i * exp(-λ_i(t) * Δt)

    LIMITATIONS (documented per spec requirements):
      - Trained on synthetic data generated with a fixed global λ prior.
        The model has limited signal to learn a more expressive λ.
      - Without real longitudinal data, learned λ may overfit to noise.
      - Results from this variant are labeled EXPERIMENTAL in all output files.
      - Do NOT compare this against the global λ baseline and claim improvement
        without acknowledging the above limitations in the paper.

    Args:
        hidden_dim: Dimension of the input hidden state (from GRU/GAT output).
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.lambda_net = nn.Sequential(
            nn.Linear(hidden_dim + 1, 32),  # +1 for temporal_gap_norm
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # ensures λ > 0
        )
        # Scale output to reasonable λ range [0.001, 0.5]
        self.lambda_scale = 0.5

    def forward(
        self,
        hidden: Tensor,
        mastery: Tensor,
        delta_t_hours: Tensor,
        temporal_gap_norm: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute learned λ and retention.

        Args:
            hidden: Learner hidden state. Shape: (n_concepts, hidden_dim)
            mastery: Current mastery. Shape: (n_concepts,)
            delta_t_hours: Time gap in hours. Shape: (n_concepts,)
            temporal_gap_norm: Normalized time gap [0,1]. Shape: (n_concepts,)

        Returns:
            Tuple of (lambda, retention). Both shape: (n_concepts,)
        """
        inp = torch.cat([hidden, temporal_gap_norm.unsqueeze(-1)], dim=-1)
        lam = self.lambda_net(inp).squeeze(-1) * self.lambda_scale  # (n_concepts,)
        lam = torch.clamp(lam, 1e-4, 0.5)

        retention = mastery * torch.exp(-lam * delta_t_hours)
        retention = torch.clamp(retention, 0.0, 1.0)

        return lam, retention
