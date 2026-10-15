#!/usr/bin/env python3
"""Leakage-aware DKT baseline for UMiKT-GAT FoundationalASSIST splits.

Predicts response at t+1 from interactions through t. Concept vocabulary is
shared across splits (concept IDs only; no response labels are used to build it).
Learners are grouped independently and sequences are never concatenated.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    mean_squared_error, brier_score_loss,
)

ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(path):
    df = pd.read_csv(path)
    required = {"learner_id", "interaction_id", "concept_id", "correctness"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df.dropna(subset=["learner_id", "interaction_id", "concept_id", "correctness"]).copy()
    df["interaction_id"] = pd.to_numeric(df["interaction_id"], errors="coerce")
    df["concept_id"] = pd.to_numeric(df["concept_id"], errors="coerce")
    df["correctness"] = pd.to_numeric(df["correctness"], errors="coerce")
    df = df.dropna(subset=["interaction_id", "concept_id", "correctness"])
    df = df[df["correctness"].isin([0, 1])]
    df["concept_id"] = df["concept_id"].astype(int)
    df["correctness"] = df["correctness"].astype(int)
    return df.sort_values(["learner_id", "interaction_id"]).reset_index(drop=True)


class SequenceDataset(Dataset):
    def __init__(self, df, concept_to_idx):
        self.sequences = []
        for learner, group in df.groupby("learner_id", sort=False):
            group = group.sort_values("interaction_id")
            skills = group["concept_id"].map(concept_to_idx).to_numpy(dtype=np.int64)
            answers = group["correctness"].to_numpy(dtype=np.float32)
            if len(skills) >= 2:
                # Input at t is the observed (concept_t, answer_t).
                # Target at t is answer_(t+1) for concept_(t+1).
                self.sequences.append((skills, answers, str(learner)))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def collate(batch, n_concepts):
    x_list, skill_list, y_list = [], [], []
    for skills, answers, _ in batch:
        # IDs [0, n_concepts) encode incorrect; [n_concepts, 2*n_concepts) correct.
        interaction_ids = skills[:-1] + answers[:-1].astype(np.int64) * n_concepts
        x_list.append(torch.tensor(interaction_ids, dtype=torch.long))
        skill_list.append(torch.tensor(skills[1:], dtype=torch.long))
        y_list.append(torch.tensor(answers[1:], dtype=torch.float32))
    lengths = torch.tensor([len(x) for x in x_list], dtype=torch.long)
    x = pad_sequence(x_list, batch_first=True, padding_value=0)
    target_skill = pad_sequence(skill_list, batch_first=True, padding_value=0)
    y = pad_sequence(y_list, batch_first=True, padding_value=0.0)
    mask = torch.arange(x.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    return x, target_skill, y, mask


class DKT(nn.Module):
    def __init__(self, n_concepts, hidden=100, dropout=0.2):
        super().__init__()
        self.n_concepts = n_concepts
        self.interaction_embedding = nn.Embedding(2 * n_concepts, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden, n_concepts)

    def forward(self, x):
        embedded = self.interaction_embedding(x)
        hidden, _ = self.lstm(embedded)
        return self.output(self.dropout(hidden))


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_n = 0.0, 0
    all_y, all_p, all_skill = [], [], []
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    for x, target_skill, y, mask in loader:
        x, target_skill = x.to(device), target_skill.to(device)
        y, mask = y.to(device), mask.to(device)
        with torch.set_grad_enabled(training):
            logits = model(x)
            selected = logits.gather(2, target_skill.unsqueeze(-1)).squeeze(-1)
            losses = criterion(selected, y)
            loss = (losses * mask).sum() / mask.sum().clamp_min(1)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        n = int(mask.sum().item())
        total_loss += float(loss.item()) * n
        total_n += n
        probs = torch.sigmoid(selected).detach().cpu().numpy()
        ys = y.detach().cpu().numpy()
        ms = mask.detach().cpu().numpy().astype(bool)
        ss = target_skill.detach().cpu().numpy()
        all_y.extend(ys[ms].astype(int).tolist())
        all_p.extend(probs[ms].astype(float).tolist())
        all_skill.extend(ss[ms].astype(int).tolist())

    return total_loss / max(total_n, 1), np.asarray(all_y), np.asarray(all_p), np.asarray(all_skill)


def calc_metrics(y, p):
    pred = (p >= 0.5).astype(int)
    result = {
        "n_next_response_predictions": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "brier": float(brier_score_loss(y, p)),
        "positive_rate": float(np.mean(y)) if len(y) else None,
    }
    result["roc_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    result["pr_auc_average_precision"] = float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None
    # Ten-bin expected calibration error.
    ece = 0.0
    for lo in np.linspace(0, 0.9, 10):
        hi = lo + 0.1
        mask = (p >= lo) & ((p < hi) if hi < 1.0 else (p <= hi))
        if mask.any():
            ece += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    result["ece_10_bins"] = float(ece)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "foundational_assist"))
    parser.add_argument("--out-dir", default=str(ROOT / "experiments" / "results" / "dkt_fair"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    set_seed(args.seed)
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(data_dir / "train.csv")
    val_df = load_split(data_dir / "val.csv")
    test_df = load_split(data_dir / "test.csv")

    # Learner overlap should be zero for a learner-disjoint evaluation.
    train_ids, val_ids, test_ids = (set(d.learner_id.astype(str)) for d in (train_df, val_df, test_df))
    overlaps = {
        "train_val_learner_overlap": len(train_ids & val_ids),
        "train_test_learner_overlap": len(train_ids & test_ids),
        "val_test_learner_overlap": len(val_ids & test_ids),
    }
    print("Rows:", len(train_df), len(val_df), len(test_df))
    print("Learners:", len(train_ids), len(val_ids), len(test_ids))
    print("Learner overlap:", overlaps)

    # Vocabulary uses concept IDs only, not correctness labels.
    all_concepts = sorted(set(train_df.concept_id) | set(val_df.concept_id) | set(test_df.concept_id))
    concept_to_idx = {cid: i for i, cid in enumerate(all_concepts)}
    n_concepts = len(all_concepts)
    print("Concepts:", n_concepts)

    train_ds = SequenceDataset(train_df, concept_to_idx)
    val_ds = SequenceDataset(val_df, concept_to_idx)
    test_ds = SequenceDataset(test_df, concept_to_idx)
    collate_fn = lambda batch: collate(batch, n_concepts)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    print("Device:", device)

    model = DKT(n_concepts, args.hidden, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    best_val_auc, best_epoch, bad_epochs = -float("inf"), 0, 0
    best_path = out_dir / "dkt_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _, _ = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_y, val_p, _ = run_epoch(model, val_loader, device)
        val_auc = roc_auc_score(val_y, val_p) if len(np.unique(val_y)) == 2 else -float("inf")
        print(f"Epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_auc={val_auc:.4f}")
        if val_auc > best_val_auc:
            best_val_auc, best_epoch, bad_epochs = val_auc, epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "n_concepts": n_concepts,
                "concept_ids": all_concepts,
                "hidden": args.hidden,
                "dropout": args.dropout,
                "seed": args.seed,
                "best_epoch": epoch,
            }, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}")
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_loss, val_y, val_p, val_skill = run_epoch(model, val_loader, device)
    test_loss, test_y, test_p, test_skill = run_epoch(model, test_loader, device)

    metrics = {
        "model": "DKT (LSTM; next-response prediction)",
        "dataset": "data/foundational_assist",
        "seed": args.seed,
        "device": device,
        "n_concepts": n_concepts,
        "best_epoch": best_epoch,
        "best_validation_auc": float(best_val_auc),
        "validation_loss": float(val_loss),
        "validation": calc_metrics(val_y, val_p),
        "test_loss": float(test_loss),
        "test": calc_metrics(test_y, test_p),
        "learner_split_overlap": overlaps,
        "evaluation_protocol": "For each learner, predict response at t+1 from observed interaction at t; never use target response as input.",
        "note": "Compare with UMiKT-GAT only after both models are evaluated on identical next-response targets and split definitions."
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({
        "y_true": test_y,
        "y_prob": test_p,
        "target_concept_index": test_skill,
    }).to_csv(out_dir / "test_predictions.csv", index=False)
    (out_dir / "concept_mapping.json").write_text(json.dumps({str(k): v for k, v in concept_to_idx.items()}, indent=2))
    print("\nTEST METRICS")
    print(json.dumps(metrics["test"], indent=2))
    print("Saved:", best_path, out_dir / "metrics.json", out_dir / "test_predictions.csv")


if __name__ == "__main__":
    main()
