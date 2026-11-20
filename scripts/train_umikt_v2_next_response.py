#!/usr/bin/env python3
"""UMiKT-GAT v2: Batched next-response training (shared GRU + GAT context).

Fixes from v1:
  - Shared GRU over full chronological sequence (not per-concept)
    → eliminates per-concept sparsity with only 7 concepts
  - Proper mini-batch training with padded sequences → ~60x faster
  - GAT on batch-averaged concept-state vectors
  - CosineAnnealing LR, patience=8
  - Identical eval protocol to DKT baseline (predict t from [0,t-1])
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    mean_squared_error,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from src.features.feature_builder import N_CONCEPTS

INTERACTION_VOCAB = 2 * N_CONCEPTS  # wrong=c, correct=c+N_CONCEPTS


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Data ──────────────────────────────────────────────────────────────────────

def load_split(path: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Returns list of (learner_id, concept_ids[T], correctness[T])."""
    df = pd.read_csv(path)
    df["correctness"] = pd.to_numeric(df["correctness"], errors="coerce")
    df["concept_id"] = pd.to_numeric(df["concept_id"], errors="coerce")
    df["interaction_id"] = pd.to_numeric(df["interaction_id"], errors="coerce")
    df = df.dropna(subset=["learner_id", "interaction_id", "concept_id", "correctness"])
    df = df[df["correctness"].isin([0.0, 1.0])]
    df["concept_id"] = df["concept_id"].astype(int) % N_CONCEPTS
    df["correctness"] = df["correctness"].astype(int)
    out = []
    for lid, g in df.groupby("learner_id", sort=False):
        g = g.sort_values("interaction_id")
        if len(g) >= 2:
            out.append((
                str(lid),
                g["concept_id"].to_numpy(dtype=np.int64),
                g["correctness"].to_numpy(dtype=np.float32),
            ))
    return out


def fit_difficulty(train_groups):
    sums = np.zeros(N_CONCEPTS)
    counts = np.zeros(N_CONCEPTS)
    for _, concepts, correctness in train_groups:
        for c, y in zip(concepts, correctness):
            sums[c] += y
            counts[c] += 1
    return np.array([(1.0 - (sums[c] + 1) / (counts[c] + 2)) for c in range(N_CONCEPTS)],
                    dtype=np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

class KTDataset(Dataset):
    """
    Each sample: (input_ids[T-1], target_concept[T-1], y[T-1], concept_counts[N_CONCEPTS])
    Input at position t = (concept[t], answer[t]) → predict concept[t+1] answer.
    """
    def __init__(self, groups, difficulty: np.ndarray):
        self.samples = []
        for _, concepts, correctness in groups:
            T = len(concepts)
            if T < 2:
                continue
            # interaction IDs: wrong=c, correct=c+N_CONCEPTS (shift by 1 to reserve 0 as pad)
            input_ids = concepts[:-1] + correctness[:-1].astype(np.int64) * N_CONCEPTS + 1
            target_c = concepts[1:]
            y = correctness[1:]
            # Cumulative concept counts from [0..T-2] prefix for each step
            # (used to build concept-state for GAT at each position)
            self.samples.append((
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_c, dtype=torch.long),
                torch.tensor(y, dtype=torch.float32),
                torch.tensor(difficulty, dtype=torch.float32),  # [N_CONCEPTS]
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    input_ids, target_c, y, diffs = zip(*batch)
    lengths = torch.tensor([len(x) for x in input_ids], dtype=torch.long)
    input_ids_pad = pad_sequence(input_ids, batch_first=True, padding_value=0)
    target_c_pad = pad_sequence(target_c, batch_first=True, padding_value=0)
    y_pad = pad_sequence(y, batch_first=True, padding_value=0.0)
    diff = torch.stack(diffs, dim=0)  # [B, N_CONCEPTS]
    max_len = input_ids_pad.size(1)
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    return input_ids_pad, target_c_pad, y_pad, mask, diff, lengths


# ── Model ──────────────────────────────────────────────────────────────────────

from src.models.umikt_gat import SharedSeqGAT, UMiKTGATv2


# ── Metrics ───────────────────────────────────────────────────────────────────

def metric_dict(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(int)
    ece = 0.0
    for lo in np.linspace(0.0, 0.9, 10):
        hi = lo + 0.1
        mask = (p >= lo) & ((p < hi) if hi < 1.0 else (p <= hi))
        if mask.any():
            ece += float(mask.mean() * abs(p[mask].mean() - y[mask].mean()))
    return {
        "n_next_response_predictions": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "pr_auc_average_precision": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else None,
        "ece_10_bins": float(ece),
        "positive_rate": float(y.mean()),
    }


@torch.no_grad()
def evaluate(model: SharedSeqGAT, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    y_all, p_all = [], []
    for input_ids, target_c, y, mask, diff, lengths in loader:
        input_ids = input_ids.to(device)
        target_c = target_c.to(device)
        diff = diff.to(device)
        lengths = lengths.to(device)
        preds = model(input_ids, target_c, diff, lengths).clamp(1e-6, 1 - 1e-6)
        m = mask.to(device)
        y_all.append(y[mask].numpy())
        p_all.append(preds[m].cpu().numpy())
    return metric_dict(np.concatenate(y_all), np.concatenate(p_all))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "foundational_assist")
    parser.add_argument("--graph-path", type=Path,
                        default=ROOT / "data" / "foundational_assist_concept_graph.json")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "experiments" / "results" / "umikt_v2_next_response")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gru-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)

    print(f"Device: {device}")
    train_raw = load_split(args.data_dir / "train.csv")
    val_raw   = load_split(args.data_dir / "val.csv")
    test_raw  = load_split(args.data_dir / "test.csv")

    train_ids = {l for l,_,_ in train_raw}
    val_ids   = {l for l,_,_ in val_raw}
    test_ids  = {l for l,_,_ in test_raw}
    overlaps = {
        "train_val":  len(train_ids & val_ids),
        "train_test": len(train_ids & test_ids),
        "val_test":   len(val_ids  & test_ids),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Learner overlap: {overlaps}")

    difficulty = fit_difficulty(train_raw)  # [N_CONCEPTS], from train only
    print(f"Difficulty per concept (train): {difficulty}")

    train_ds = KTDataset(train_raw, difficulty)
    val_ds   = KTDataset(val_raw,   difficulty)
    test_ds  = KTDataset(test_raw,  difficulty)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    model = SharedSeqGAT(
        H=args.hidden_dim, n_heads=args.n_heads, dropout=args.dropout,
        graph_path=str(args.graph_path), gru_layers=args.gru_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Learners — train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    print(f"Batches/epoch: {len(train_loader)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.BCELoss()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_auc = -float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for input_ids, target_c, y, mask, diff, lengths in train_loader:
            input_ids = input_ids.to(device)
            target_c  = target_c.to(device)
            y         = y.to(device)
            diff      = diff.to(device)
            lengths   = lengths.to(device)
            mask_d    = mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            preds = model(input_ids, target_c, diff, lengths).clamp(1e-6, 1 - 1e-6)
            loss = criterion(preds[mask_d], y[mask_d])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        val_auc = val_metrics["roc_auc"] or -float("inf")
        print(
            f"Epoch {epoch:03d} | loss={np.mean(losses):.4f} | "
            f"val_auc={val_auc:.4f} | val_rmse={val_metrics['rmse']:.4f} | "
            f"elapsed={time.time()-start:.0f}s"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
            torch.save({
                "model_state_dict": best_state,
                "epoch": epoch,
                "val_metrics": val_metrics,
                "args": vars(args),
            }, args.out_dir / "umikt_v2_best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError("No checkpoint saved — training failed.")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    test_metrics.update({
        "best_epoch": best_epoch,
        "best_validation_auc": float(best_auc),
        "elapsed_seconds": round(time.time() - start, 2),
        "n_params": n_params,
        "learner_overlap": overlaps,
        "model": "UMiKT-GAT v2 (SharedSeqGAT: batched GRU + GAT concept context)",
        "evaluation_protocol": (
            "Predict correctness at position t from observed interactions [0,t-1]. "
            "Identical protocol to DKT baseline (experiments/results/dkt_fair/metrics.json)."
        ),
    })

    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("\n=== Test Metrics ===")
    print(json.dumps(test_metrics, indent=2))

    # Side-by-side comparison vs DKT
    dkt_path = ROOT / "experiments" / "results" / "dkt_fair" / "metrics.json"
    if dkt_path.exists():
        with open(dkt_path) as f:
            dkt = json.load(f)
        dkt_test = dkt.get("test", dkt)
        print("\n=== UMiKT-GAT v2 vs DKT (identical evaluation protocol) ===")
        print(f"{'Metric':<35} {'UMiKT-v2':>12} {'DKT':>12} {'Delta':>10}")
        print("-" * 72)
        for m in ["roc_auc", "pr_auc_average_precision", "rmse", "brier", "accuracy", "ece_10_bins"]:
            um = test_metrics.get(m)
            dk = dkt_test.get(m)
            if um is not None and dk is not None:
                delta = um - dk
                marker = "✓" if (m in ["roc_auc", "pr_auc_average_precision", "accuracy"] and delta > 0) or \
                               (m in ["rmse", "brier", "ece_10_bins"] and delta < 0) else " "
                print(f"  {marker}{m:<33} {um:>12.4f} {dk:>12.4f} {delta:>+10.4f}")

    print(f"\nSaved: {args.out_dir}")


if __name__ == "__main__":
    main()
