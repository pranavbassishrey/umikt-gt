#!/usr/bin/env python3
"""Leakage-controlled next-response training for UMiKT-GAT.

This is a separate experiment script. It does not modify the original benchmark.
For each target interaction t, the model sees only the learner's records before t
and predicts correctness for the target interaction's concept.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    mean_squared_error,
    roc_auc_score,
)


import sys

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from src.features.feature_builder import build_feature_vector, INPUT_DIM, N_CONCEPTS
from src.models.umikt_gat import UMiKTGATModel, make_default_config

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_records(path: Path) -> list[dict]:
    df = pd.read_csv(path)
    if "response_time_sec" in df:
        df["response_time_norm"] = (
            np.log1p(pd.to_numeric(df["response_time_sec"], errors="coerce").fillna(0).clip(lower=0))
            / np.log1p(600.0)
        ).clip(0, 1)
    if "attempts" in df:
        df["attempts"] = (
            pd.to_numeric(df["attempts"], errors="coerce").fillna(1).div(5.0).clip(0, 1)
        )
    if "temporal_gap_hours" in df:
        df["temporal_gap_norm"] = (
            pd.to_numeric(df["temporal_gap_hours"], errors="coerce").fillna(0).clip(lower=0)
            / 24.0
        ).clip(0, 1)
    records = df.to_dict(orient="records")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        grouped[str(row["learner_id"])].append(row)
    ordered = []
    for learner_id in sorted(grouped):
        seq = grouped[learner_id]
        seq.sort(key=lambda r: int(float(r.get("interaction_id", 0) or 0)))
        ordered.append((learner_id, seq))
    return ordered


def concept_id(row: dict) -> int:
    # Match the repository's existing benchmark convention.
    return int(float(row.get("concept_id", 0) or 0)) % N_CONCEPTS


def correctness(row: dict) -> int:
    return int(float(row.get("correctness", 0) or 0) >= 0.5)


def fit_training_difficulty(train_groups: list[tuple[str, list[dict]]]) -> dict[int, float]:
    """Estimate concept difficulty from TRAIN labels only, with smoothing."""
    sums = np.zeros(N_CONCEPTS, dtype=np.float64)
    counts = np.zeros(N_CONCEPTS, dtype=np.float64)
    for _, seq in train_groups:
        for row in seq:
            c = concept_id(row)
            sums[c] += correctness(row)
            counts[c] += 1
    # Beta(1,1) smoothing; difficulty = 1 - smoothed training accuracy.
    return {c: float(1.0 - (sums[c] + 1.0) / (counts[c] + 2.0)) for c in range(N_CONCEPTS)}


def apply_train_difficulty(groups, difficulty_by_concept):
    # Do not let a globally precomputed difficulty field leak validation/test labels.
    for _, seq in groups:
        for row in seq:
            row["difficulty"] = difficulty_by_concept[concept_id(row)]


def learner_states(prefix: list[dict], device: torch.device) -> torch.Tensor:
    """Construct [M, MC, U, R] from prefix history only."""
    counts = np.zeros(N_CONCEPTS, dtype=np.float32)
    sums = np.zeros(N_CONCEPTS, dtype=np.float32)
    last_pos = np.full(N_CONCEPTS, -10_000, dtype=np.int64)
    for pos, row in enumerate(prefix):
        c = concept_id(row)
        counts[c] += 1
        sums[c] += correctness(row)
        last_pos[c] = pos

    states = np.zeros((N_CONCEPTS, 4), dtype=np.float32)
    for c in range(N_CONCEPTS):
        n = counts[c]
        states[c, 0] = (sums[c] / n) if n else 0.5       # observed mastery proxy
        states[c, 1] = 0.0                              # no misconception labels available
        states[c, 2] = 1.0 / math.sqrt(n + 1.0)         # uncertainty proxy
        states[c, 3] = 1.0 / (1.0 + len(prefix) - 1 - last_pos[c]) if n else 0.0
    return torch.tensor(states, dtype=torch.float32, device=device)


def make_input(prefix: list[dict], seq_len: int, device: torch.device):
    x = torch.zeros((N_CONCEPTS, seq_len, INPUT_DIM), dtype=torch.float32, device=device)
    by_concept: dict[int, list[dict]] = defaultdict(list)
    for row in prefix:
        by_concept[concept_id(row)].append(row)
    for c, seq in by_concept.items():
        # Use the most recent history for each concept, preserving chronology.
        seq = seq[-seq_len:]
        offset = seq_len - len(seq)
        for j, row in enumerate(seq):
            fv = build_feature_vector(row)
            x[c, offset + j] = torch.as_tensor(fv, dtype=torch.float32, device=device)
    return x


def examples(groups, stride: int = 1):
    """Each example predicts seq[t] from seq[:t], never including target t."""
    for learner_id, seq in groups:
        # t=1 means the learner has at least one observed interaction.
        for t in range(1, len(seq), max(1, stride)):
            yield learner_id, seq, t


@torch.no_grad()
def evaluate(model, groups, device, seq_len, stride=1):
    model.eval()
    y_true, y_prob = [], []
    for _, seq, t in examples(groups, stride):
        prefix = seq[:t]
        target = seq[t]
        x = make_input(prefix, seq_len, device)
        states = learner_states(prefix, device)
        out = model(x, states)
        c = concept_id(target)
        p = float(out["mastery"][c].clamp(1e-6, 1 - 1e-6).item())
        y_true.append(correctness(target))
        y_prob.append(p)
    return metric_dict(np.asarray(y_true, dtype=int), np.asarray(y_prob, dtype=float))


def metric_dict(y, p):
    if len(y) == 0:
        raise RuntimeError("No evaluation targets were produced.")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "foundational_assist")
    parser.add_argument("--graph-path", type=Path, default=ROOT / "data" / "foundational_assist_concept_graph.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments" / "results" / "umikt_next_response")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seq-len", type=int, default=15)
    parser.add_argument("--train-stride", type=int, default=3,
                        help="Use every Nth training target to reduce runtime; validation/test use every target.")
    parser.add_argument("--max-train-examples", type=int, default=12000,
                        help="Cap examples per epoch; 0 means no cap.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    train = load_records(args.data_dir / "train.csv")
    val = load_records(args.data_dir / "val.csv")
    test = load_records(args.data_dir / "test.csv")

    train_ids = {lid for lid, _ in train}
    val_ids = {lid for lid, _ in val}
    test_ids = {lid for lid, _ in test}
    overlaps = {
        "train_val": len(train_ids & val_ids),
        "train_test": len(train_ids & test_ids),
        "val_test": len(val_ids & test_ids),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Learner overlap across splits: {overlaps}")

    difficulty = fit_training_difficulty(train)
    apply_train_difficulty(train, difficulty)
    apply_train_difficulty(val, difficulty)
    apply_train_difficulty(test, difficulty)

    train_examples = list(examples(train, args.train_stride))
    rng = random.Random(args.seed)
    if args.max_train_examples > 0 and len(train_examples) > args.max_train_examples:
        train_examples = rng.sample(train_examples, args.max_train_examples)

    config = make_default_config()
    config.update({
        "hidden_dim": 64,
        "gru_layers": 2,
        "n_heads": 4,
        "dropout_rate": 0.30,
        "gat_dropout": 0.20,
        "graph_path": str(args.graph_path),
        "use_graph": True,
        "dynamic_gat": True,
        "use_misconception_head": False,
        "use_retention": False,
    })
    model = UMiKTGATModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_auc = -float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    start = time.time()

    print(f"Device: {device}")
    print(f"Split learners: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    print(f"Split overlap: {overlaps}")
    print(f"Training targets/epoch: {len(train_examples)} (stride={args.train_stride})")
    print("Validation evaluates every next-response target; test is not used for model selection.")

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_examples)
        losses = []
        for _, seq, t in train_examples:
            prefix, target = seq[:t], seq[t]
            x = make_input(prefix, args.seq_len, device)
            states = learner_states(prefix, device)
            c = concept_id(target)
            y = torch.tensor(float(correctness(target)), dtype=torch.float32, device=device)

            optimizer.zero_grad(set_to_none=True)
            out = model(x, states)
            p = out["mastery"][c].clamp(1e-6, 1 - 1e-6)
            loss = criterion(p, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))

        val_metrics = evaluate(model, val, device, args.seq_len, stride=1)
        val_auc = val_metrics["roc_auc"]
        score = val_auc if val_auc is not None else -float("inf")
        print(
            f"Epoch {epoch:03d} | loss={np.mean(losses):.4f} | "
            f"val_auc={score:.4f} | val_rmse={val_metrics['rmse']:.4f}"
        )
        if score > best_auc:
            best_auc = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
            torch.save({
                "model_state_dict": best_state,
                "config": config,
                "epoch": epoch,
                "val_metrics": val_metrics,
                "seed": args.seed,
            }, args.out_dir / "umikt_next_response_best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training failed to produce a checkpoint.")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test, device, args.seq_len, stride=1)
    test_metrics["best_epoch"] = best_epoch
    test_metrics["best_validation_auc"] = float(best_auc)
    test_metrics["elapsed_seconds"] = round(time.time() - start, 2)
    test_metrics["learner_overlap"] = overlaps
    test_metrics["train_targets_per_epoch"] = len(train_examples)
    test_metrics["train_stride"] = args.train_stride
    test_metrics["difficulty_estimated_from"] = "training split only"

    # Export per-target predictions for paired comparison against DKT.
    model.eval()
    rows = []
    with torch.no_grad():
        for learner_id, seq, t in examples(test, 1):
            prefix, target = seq[:t], seq[t]
            out = model(make_input(prefix, args.seq_len, device), learner_states(prefix, device))
            c = concept_id(target)
            rows.append({
                "learner_id": learner_id,
                "target_interaction_id": target.get("interaction_id", t),
                "target_index_in_learner": t,
                "concept_id_mapped": c,
                "y_true": correctness(target),
                "y_prob": float(out["mastery"][c].clamp(1e-6, 1 - 1e-6).item()),
            })

    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    pd.DataFrame(rows).to_csv(args.out_dir / "test_predictions.csv", index=False)
    with open(args.out_dir / "concept_mapping.json", "w") as f:
        json.dump({
            "n_concepts": N_CONCEPTS,
            "mapping": "int(concept_id) % N_CONCEPTS (matches current repository convention)",
        }, f, indent=2)

    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=2))
    print(f"Saved results to: {args.out_dir}")


if __name__ == "__main__":
    main()
