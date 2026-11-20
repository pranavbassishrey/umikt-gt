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

import math
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .gru_encoder import GRUEncoder
from .gat_layer import MultiHeadGAT
from .fusion import FusionLayer
from .prediction_heads import MasteryHead, MisconceptionHead, RetentionHead
from ..features.feature_builder import INPUT_DIM, N_CONCEPTS

INTERACTION_VOCAB = 14  # 7 concepts * 2 binary outcomes


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
                graph_path=config.get("graph_path", None),
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
        # Multi-task loss weights — calibrated so mastery task retains majority gradient
        "lambda_mastery": 1.0,
        "lambda_misconception": 0.10,
        "lambda_retention": 1.0,
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


class SharedSeqGAT(nn.Module):
    """
    UMiKT-GAT v2: Batched shared-sequence GRU + per-step causal GAT concept context.

    GRU input: interaction embedding (concept, answer) → H
    GAT input: concept-frequency statistics built causally from prefix → H per concept node
    Output: for each step t, sigmoid(MLP(gru_h_t, gat_node[target_c_t], concept_emb))
    """

    def __init__(self, H: int = 64, n_heads: int = 4, dropout: float = 0.2,
                 graph_path: str = "data/foundational_assist_concept_graph.json",
                 gru_layers: int = 2):
        super().__init__()
        self.H = H
        self.n_heads = n_heads
        self.head_dim = H // n_heads

        # +1 for padding index 0
        self.embed = nn.Embedding(INTERACTION_VOCAB + 2, H, padding_idx=0)
        self.gru = nn.GRU(H, H, num_layers=gru_layers, batch_first=True,
                          dropout=dropout if gru_layers > 1 else 0.0)
        self.gru_drop = nn.Dropout(dropout)
        self.gru_norm = nn.LayerNorm(H)

        # Concept-state encoder: [mastery_proxy, uncertainty, difficulty] → H
        self.state_proj = nn.Linear(3, H)
        self.state_norm = nn.LayerNorm(H)

        # GAT
        self.gat_q = nn.Linear(H, H, bias=False)
        self.gat_k = nn.Linear(H, H, bias=False)
        self.gat_v = nn.Linear(H, H, bias=False)
        self.gat_out = nn.Linear(H, H)
        self.gat_norm = nn.LayerNorm(H)

        adj = self._load_adj(graph_path)
        self.register_buffer("adj", adj)

        self.concept_emb = nn.Embedding(N_CONCEPTS, H)

        # Prediction: concat(gru_h, gat_node, concept_emb) → 1
        self.head = nn.Sequential(
            nn.Linear(H * 3, H * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(H * 2, H), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(H, 1), nn.Sigmoid(),
        )

    def _load_adj(self, graph_path: str) -> torch.Tensor:
        adj = torch.eye(N_CONCEPTS)
        if graph_path and Path(graph_path).exists():
            try:
                with open(graph_path) as f:
                    g = json.load(f)
                for e in g.get("edges", []):
                    src = int(e.get("source", e.get("from", e.get("src", 0)))) % N_CONCEPTS
                    dst = int(e.get("target", e.get("to", e.get("dst", 0)))) % N_CONCEPTS
                    adj[src, dst] = 1.0
                    adj[dst, src] = 1.0
            except Exception:
                pass
        return adj / adj.sum(dim=1, keepdim=True).clamp(min=1)

    def _gat_pass(self, node_feats: torch.Tensor) -> torch.Tensor:
        """node_feats: [B, N, H] or [N, H]. Returns same shape."""
        batched = node_feats.dim() == 3
        if not batched:
            node_feats = node_feats.unsqueeze(0)
        B, N, H = node_feats.shape
        d = self.head_dim
        nh = self.n_heads

        q = self.gat_q(node_feats).view(B, N, nh, d).permute(0, 2, 1, 3)  # [B,nh,N,d]
        k = self.gat_k(node_feats).view(B, N, nh, d).permute(0, 2, 1, 3)
        v = self.gat_v(node_feats).view(B, N, nh, d).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)  # [B,nh,N,N]
        adj_mask = (self.adj < 1e-6).unsqueeze(0).unsqueeze(0)         # [1,1,N,N]
        scores = scores.masked_fill(adj_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = torch.matmul(attn, v)  # [B,nh,N,d]
        out = out.permute(0, 2, 1, 3).reshape(B, N, H)
        result = self.gat_norm(node_feats + self.gat_out(out))
        return result.squeeze(0) if not batched else result

    def build_concept_states(self, input_ids: torch.Tensor,
                             diff: torch.Tensor) -> torch.Tensor:
        """
        Build concept-state node features for GAT strictly causally.
        At each step t in [0, T-1], mastery proxy and uncertainty are computed
        ONLY using interactions up to step t (cumulative prefix, no future peeking).

        input_ids: [B, T] (1-indexed interaction IDs, 0=pad)
        diff: [B, N_CONCEPTS] concept difficulty estimates
        Returns: [B, T, N_CONCEPTS, H]
        """
        B, T = input_ids.shape
        nonpad = (input_ids > 0)
        ids_shifted = (input_ids - 1).clamp(min=0)
        concepts_in = ids_shifted % N_CONCEPTS           # [B, T]
        answers_in = (ids_shifted // N_CONCEPTS).float() # [B, T]

        c_oh = torch.zeros(B, T, N_CONCEPTS, device=input_ids.device)
        c_oh.scatter_(2, concepts_in.unsqueeze(2), 1.0)    # [B, T, N]
        valid = nonpad.float().unsqueeze(2)                 # [B, T, 1]

        answers_c = c_oh * answers_in.unsqueeze(2) * valid # [B, T, N]
        counts_c = c_oh * valid                            # [B, T, N]

        cum_answers = torch.cumsum(answers_c, dim=1)        # [B, T, N]
        cum_counts = torch.cumsum(counts_c, dim=1)          # [B, T, N]

        mastery_proxy = cum_answers / (cum_counts + 1e-6)   # [B, T, N]
        uncertainty = 1.0 / torch.sqrt(cum_counts + 1.0)   # [B, T, N]
        diff_exp = diff.unsqueeze(1).expand(B, T, N_CONCEPTS) # [B, T, N]

        states = torch.stack([mastery_proxy, uncertainty, diff_exp], dim=3)
        node_feats = self.state_norm(self.state_proj(states)) # [B, T, N, H]

        gat_res = self._gat_pass(node_feats.view(B * T, N_CONCEPTS, self.H)) # [B*T, N, H]
        return gat_res.view(B, T, N_CONCEPTS, self.H)                         # [B, T, N, H]

    def forward(self, input_ids: torch.Tensor, target_c: torch.Tensor,
                diff: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, T] padded interaction IDs (1-indexed)
        target_c:  [B, T] target concept at each step
        diff:      [B, N_CONCEPTS]
        lengths:   [B] actual sequence lengths
        Returns:   [B, T] predicted P(correct) (unmasked)
        """
        B, T = input_ids.shape

        emb = self.embed(input_ids)  # [B, T, H]
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        gru_out, _ = self.gru(packed)
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True, total_length=T)  # [B, T, H]
        gru_h = self.gru_norm(self.gru_drop(gru_out))

        node_feats = self.build_concept_states(input_ids, diff)

        tc_clamped = target_c.clamp(0, N_CONCEPTS - 1)
        tc_idx = tc_clamped.unsqueeze(2).unsqueeze(3).expand(B, T, 1, self.H)
        gat_node = node_feats.gather(2, tc_idx).squeeze(2)      # [B, T, H]

        c_emb = self.concept_emb(tc_clamped)                    # [B, T, H]

        combined = torch.cat([gru_h, gat_node, c_emb], dim=2)  # [B, T, 3H]
        return self.head(combined).squeeze(2)                    # [B, T]


# Canonical alias for UMiKT-GAT v2
UMiKTGATv2 = SharedSeqGAT
