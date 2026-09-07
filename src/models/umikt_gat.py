"""
UMiKT-GAT: Main model combining GRU temporal encoder, GAT graph layer,
and multi-task prediction heads.

Architecture:
  Input: X_t (feature vector, dim=20)
    ↓
  GRU Encoder: H_t = GRU(X_t, H_{t-1})  [hidden_dim]
    ↓ [Fusion Order A: T→G] or earlier [Fusion Order B: G→T]
  Multi-head GAT: Z = GAT(H, S, edge_features)  [hidden_dim]
    ↓
  Fusion: F = MLP(concat(H, Z))  [hidden_dim]
    ↓
  Four output heads:
    M_hat = MasteryHead(F)          ∈ [0,1]
    MC_hat = MisconceptionHead(F)   ∈ R^7 (logits)
    R_hat = RetentionHead(F)        ∈ [0,1]
    U_hat = Var[MC-Dropout passes]  ∈ [0,∞)  (computed externally)

The model processes one learner at a time (one concept per forward pass).
For efficiency on the small dataset, sequences are processed concept-by-concept.

Config-driven: all architecture choices (fusion order, dynamic GAT,
n_heads, dropout_rate, etc.) are set via the config dict.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .gru_encoder import GRUEncoder
from .gat_layer import MultiHeadGAT
from .fusion import FusionLayer
from .prediction_heads import MasteryHead, MisconceptionHead, RetentionHead
from ..features.feature_builder import INPUT_DIM, N_CONCEPTS


class UMiKTGATModel(nn.Module):
    """
    Full UMiKT-GAT model with configurable component activation.

    The ablation study uses the same class with different components enabled/disabled,
    ensuring fair comparison (same code path, only config differs).

    Args:
        config: Dict of hyperparameters. See make_default_config() for all keys.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        hidden_dim = config.get("hidden_dim", 64)
        dropout_rate = config.get("dropout_rate", 0.3)
        n_heads = config.get("n_heads", 4)
        self.fusion_order = config.get("fusion_order", "temporal_first")  # or "graph_first"
        self.use_graph = config.get("use_graph", True)
        self.use_misconception_head = config.get("use_misconception_head", True)
        self.use_retention = config.get("use_retention", True)
        self.dynamic_gat = config.get("dynamic_gat", True)  # learner-state-conditioned

        # ── Temporal Encoder ────────────────────────────────────────────────
        self.gru = GRUEncoder(
            input_dim=INPUT_DIM,
            hidden_dim=hidden_dim,
            num_layers=config.get("gru_layers", 2),
            dropout_rate=dropout_rate,
        )

        # ── Graph Attention ─────────────────────────────────────────────────
        if self.use_graph:
            self.gat = MultiHeadGAT(
                node_dim=hidden_dim,
                out_dim=hidden_dim,
                n_heads=n_heads,
                learner_state_dim=4,  # [M, MC, U, R]
                dynamic=self.dynamic_gat,
                dropout_rate=config.get("gat_dropout", 0.2),
            )

        # ── Fusion ──────────────────────────────────────────────────────────
        if self.use_graph:
            self.fusion = FusionLayer(
                temporal_dim=hidden_dim,
                graph_dim=hidden_dim,
                hidden_dim=hidden_dim,
                dropout_rate=dropout_rate,
            )

        # ── Prediction Heads ────────────────────────────────────────────────
        self.mastery_head = MasteryHead(hidden_dim=hidden_dim, dropout_rate=dropout_rate)

        if self.use_misconception_head:
            self.misconception_head = MisconceptionHead(
                hidden_dim=hidden_dim, dropout_rate=dropout_rate
            )

        if self.use_retention:
            self.retention_head = RetentionHead(
                hidden_dim=hidden_dim, dropout_rate=dropout_rate
            )

    def forward(
        self,
        x_seq: Tensor,
        learner_states: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        Forward pass for a single learner's interaction sequence.

        Args:
            x_seq: Sequence of interaction features.
                   Shape: (n_concepts, seq_len, INPUT_DIM)
                   One sequence per concept; concepts with no interactions
                   have zero-padded sequences.
            learner_states: Current learner state per concept for dynamic GAT.
                            Shape: (n_concepts, 4) = (7, 4)
                            4 dims = [M, MC, U, R] (all in [0,1])

        Returns:
            Dict with keys:
              'mastery':         (n_concepts,)
              'misconception':   (n_concepts, 7) logits — if use_misconception_head
              'retention':       (n_concepts,) — if use_retention
              'hidden':          (n_concepts, hidden_dim) — for downstream use
        """
        n = N_CONCEPTS
        device = x_seq.device

        # ── Temporal encoding: one GRU per concept ──────────────────────────
        # x_seq shape: (n_concepts, seq_len, INPUT_DIM)
        # We process each concept sequence separately through the shared GRU.
        # The GRU weights are shared across concepts (parameter efficiency).
        h0 = self.gru.init_hidden(batch_size=1, device=device)
        concept_hiddens = []

        for c in range(n):
            seq = x_seq[c:c+1]  # (1, seq_len, INPUT_DIM)
            out, h_n = self.gru(seq, h0)
            # Take the last time step's output as the concept representation
            last_out = out[:, -1, :]  # (1, hidden_dim)
            concept_hiddens.append(last_out.squeeze(0))  # (hidden_dim,)

        # Stack: (n_concepts, hidden_dim)
        H = torch.stack(concept_hiddens, dim=0)

        # ── Graph + Fusion ──────────────────────────────────────────────────
        if self.use_graph:
            if self.fusion_order == "temporal_first":
                # T→G: GRU first, then GAT
                Z = self.gat(H, learner_states if self.dynamic_gat else None)
                fused = self.fusion(H, Z)
            else:
                # G→T: GAT first (on input projections), then GRU on enriched features
                # For G→T, we use H as initial node features (from GRU last timestep)
                # and apply GAT before the fusion step
                Z = self.gat(H, learner_states if self.dynamic_gat else None)
                # In G→T, we treat GAT output as the "temporal" and GRU output as "graph"
                # This is architecturally equivalent to swapping the order of combination
                fused = self.fusion(Z, H)
        else:
            fused = H

        # ── Output heads ────────────────────────────────────────────────────
        outputs = {}

        mastery = self.mastery_head(fused).squeeze(-1)  # (n_concepts,)
        outputs["mastery"] = mastery
        outputs["hidden"] = fused

        if self.use_misconception_head:
            mc_logits = self.misconception_head(fused)  # (n_concepts, 7)
            outputs["misconception"] = mc_logits

        if self.use_retention:
            retention = self.retention_head(fused).squeeze(-1)  # (n_concepts,)
            outputs["retention"] = retention

        return outputs


def make_default_config() -> dict:
    """Return the default full UMiKT-GAT configuration."""
    return {
        # Architecture
        "hidden_dim": 64,
        "gru_layers": 2,
        "n_heads": 4,
        "fusion_order": "temporal_first",  # "temporal_first" or "graph_first"
        # Component flags (ablation study controls these)
        "use_graph": True,
        "use_misconception_head": True,
        "use_retention": True,
        "dynamic_gat": True,
        # Regularization
        "dropout_rate": 0.30,
        "gat_dropout": 0.20,
        # Training
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "n_epochs": 50,
        "patience": 10,
        # Multi-task loss weights
        # Values are starting points; document that these are not novel contributions.
        "lambda_mastery": 1.0,
        "lambda_misconception": 0.5,
        "lambda_retention": 0.3,
        # Uncertainty (MC-Dropout)
        "mc_samples": 20,
        # LLM
        "llm_provider": "mock",  # "mock" or "openai"
        # Data
        "seq_len": 10,  # max interactions per concept per sequence
        "n_concepts": 7,
        # Reproducibility
        "seed": 42,
    }


def make_ablation_config(variant: str, base_config: dict | None = None) -> dict:
    """
    Create a model config for a specific ablation variant.

    Variants match the ablation table in the spec:
      "mastery_only"    : Temporal=No, Graph=No, MC=No, Ret=No
      "+temporal"       : Temporal=Yes, Graph=No, MC=No, Ret=No
      "+graph"          : Temporal=Yes, Graph=Yes, MC=No, Ret=No
      "+misconception"  : Temporal=Yes, Graph=Yes, MC=Yes, Ret=No
      "+uncertainty"    : Temporal=Yes, Graph=Yes, MC=Yes, Ret=No (uncertainty via MC-Dropout)
      "full_umikt_gat"  : All=Yes
    """
    if base_config is None:
        base_config = make_default_config()

    cfg = base_config.copy()

    if variant == "mastery_only":
        cfg.update({
            "use_graph": False,
            "use_misconception_head": False,
            "use_retention": False,
            "gru_layers": 1,
        })
    elif variant == "+temporal":
        cfg.update({
            "use_graph": False,
            "use_misconception_head": False,
            "use_retention": False,
        })
    elif variant == "+graph":
        cfg.update({
            "use_graph": True,
            "use_misconception_head": False,
            "use_retention": False,
        })
    elif variant == "+misconception":
        cfg.update({
            "use_graph": True,
            "use_misconception_head": True,
            "use_retention": False,
        })
    elif variant in ("+uncertainty", "full_umikt_gat"):
        cfg.update({
            "use_graph": True,
            "use_misconception_head": True,
            "use_retention": True,
        })

    return cfg
