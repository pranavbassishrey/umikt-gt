"""
UMiKT-GAT Training Script.

Trains a single model variant defined by a config dict or YAML file.
All hyperparameters are logged. Evaluation metrics are computed and saved.

Usage:
  python src/train.py                         # train full model with default config
  python src/train.py --config experiments/configs/phase3_mastery_baseline.yaml
  python src/train.py --variant +temporal    # quick ablation variant

Metrics saved to experiments/results/{variant_name}/train_metrics.json
"""

import argparse
import json
import os
import sys
import time
import random
import math
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np

from src.data.loader import load_synthetic_split, group_by_learner
from src.features.feature_builder import build_feature_vector, build_labels, N_CONCEPTS, INPUT_DIM
from src.models.umikt_gat import UMiKTGATModel, make_default_config, make_ablation_config
from src.models.retention_module import ExponentialDecayRetention


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_sequences(
    learner_records: list[dict],
    seq_len: int = 10,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Build input sequence tensor and label tensors for one learner.

    Groups interactions by concept, builds a sequence of feature vectors
    per concept (zero-padded to seq_len), and extracts labels.

    Returns:
        x_seq: Shape (N_CONCEPTS, seq_len, INPUT_DIM)
        labels: Dict of label tensors per concept.
    """
    # Group by concept
    by_concept = defaultdict(list)
    for r in learner_records:
        cid = int(r.get("concept_id", 0))
        by_concept[cid].append(r)

    x_seq = torch.zeros(N_CONCEPTS, seq_len, INPUT_DIM, device=device)
    labels_mastery = torch.zeros(N_CONCEPTS, device=device)
    labels_mc = torch.zeros(N_CONCEPTS, dtype=torch.long, device=device)
    labels_ret = torch.zeros(N_CONCEPTS, device=device)
    temporal_gaps = torch.zeros(N_CONCEPTS, device=device)
    active_mask = torch.zeros(N_CONCEPTS, dtype=torch.bool, device=device)

    for cid in range(N_CONCEPTS):
        recs = by_concept.get(cid, [])
        for t, rec in enumerate(recs[:seq_len]):
            fv = build_feature_vector(rec)
            x_seq[cid, t] = torch.tensor(fv, dtype=torch.float32, device=device)

        if recs:
            active_mask[cid] = True
            last_rec = recs[-1]
            lab = build_labels(last_rec)
            labels_mastery[cid] = lab["mastery"]
            labels_mc[cid] = lab["misconception_class"]
            labels_ret[cid] = lab["retention"]
            temporal_gaps[cid] = float(recs[-1].get("temporal_gap_norm", 0.1))

    return x_seq, {
        "mastery": labels_mastery,
        "misconception": labels_mc,
        "retention": labels_ret,
        "temporal_gaps": temporal_gaps,
        "active_mask": active_mask,
    }


def compute_class_weights(records: list[dict], n_classes: int = 7) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for misconception head.

    Classes with fewer samples get higher weights to address imbalance.
    """
    counts = torch.zeros(n_classes)
    for r in records:
        mc = int(r.get("gt_misconception_class", 0))
        if 0 <= mc < n_classes:
            counts[mc] += 1
    total = counts.sum()
    if total == 0:
        return torch.ones(n_classes)
    freq = counts / total
    weights = 1.0 / (freq + 1e-6)
    weights = weights / weights.mean()  # normalize so mean weight = 1
    return weights


def train_epoch(
    model: UMiKTGATModel,
    records_by_learner: dict,
    optimizer: torch.optim.Optimizer,
    config: dict,
    device: torch.device,
    mastery_loss_fn: nn.Module,
    mc_loss_fn: nn.Module | None,
    retention_loss_fn: nn.Module | None,
) -> dict:
    """Train for one epoch. Returns dict of average losses."""
    model.train()
    total_loss = total_mastery_loss = total_mc_loss = total_ret_loss = 0.0
    n_batches = 0

    seq_len = config.get("seq_len", 10)
    lam1 = config.get("lambda_mastery", 1.0)
    lam2 = config.get("lambda_misconception", 0.5)
    lam3 = config.get("lambda_retention", 0.3)

    learner_ids = list(records_by_learner.keys())
    random.shuffle(learner_ids)

    for lid in learner_ids:
        recs = records_by_learner[lid]
        if not recs:
            continue

        x_seq, labels = build_sequences(recs, seq_len=seq_len, device=device)

        # Build initial learner state for dynamic GAT
        # IMPORTANT FIX: Do NOT pass target mastery_gt into learner_states input (that causes data leakage!)
        # Pass realistic prior/estimated state [M=0.3, MC=0.0, U=0.5, R=0.8]
        prior_m = torch.full((N_CONCEPTS,), 0.3, device=device)
        prior_r = torch.full((N_CONCEPTS,), 0.8, device=device)
        learner_states = torch.stack([
            prior_m,                                    # M prior
            torch.zeros(N_CONCEPTS, device=device),     # MC
            torch.full((N_CONCEPTS,), 0.5, device=device), # U
            prior_r,                                    # R prior
        ], dim=1)  # (N_CONCEPTS, 4)

        outputs = model(x_seq, learner_states)

        # Sanity check logging on first batch of first epoch
        if n_batches == 0 and not hasattr(train_epoch, "_logged_sanity"):
            train_epoch._logged_sanity = True
            feat_min, feat_max, feat_mean = x_seq.min().item(), x_seq.max().item(), x_seq.mean().item()
            raw_logits = model.mastery_head.net[:-1](outputs["hidden"]).squeeze(-1) if hasattr(model.mastery_head, "net") else outputs["mastery"]
            print(f"\n  [SANITY CHECK - Train Batch 0]")
            print(f"    x_seq tensor stats  : min={feat_min:.3f}, max={feat_max:.3f}, mean={feat_mean:.3f}, shape={list(x_seq.shape)}")
            print(f"    Raw mastery logits  : min={raw_logits.min().item():.3f}, max={raw_logits.max().item():.3f}, mean={raw_logits.mean().item():.3f}")
            print(f"    Post-sigmoid mastery: min={outputs['mastery'].min().item():.3f}, max={outputs['mastery'].max().item():.3f}, mean={outputs['mastery'].mean().item():.3f}")

        mask = labels["active_mask"]
        if not mask.any():
            continue

        # ── Mastery loss ────────────────────────────────────────────────────
        pred_mastery = outputs["mastery"][mask]
        L_mastery = mastery_loss_fn(pred_mastery, labels["mastery"][mask])

        loss = lam1 * L_mastery
        total_mastery_loss += L_mastery.item()

        # ── Misconception loss ──────────────────────────────────────────────
        if mc_loss_fn is not None and "misconception" in outputs:
            pred_mc = outputs["misconception"][mask]  # (N_active, 7)
            L_mc = mc_loss_fn(pred_mc, labels["misconception"][mask])
            loss = loss + lam2 * L_mc
            total_mc_loss += L_mc.item()

        # ── Retention loss ──────────────────────────────────────────────────
        if retention_loss_fn is not None and "retention" in outputs:
            pred_ret = outputs["retention"][mask]  # (N_active,)
            L_ret = retention_loss_fn(pred_ret, labels["retention"][mask])
            loss = loss + lam3 * L_ret
            total_ret_loss += L_ret.item()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if n_batches == 0:
        return {"total": 0, "mastery": 0, "mc": 0, "ret": 0}

    return {
        "total": total_loss / n_batches,
        "mastery": total_mastery_loss / n_batches,
        "mc": total_mc_loss / n_batches,
        "ret": total_ret_loss / n_batches,
    }


@torch.no_grad()
def evaluate(
    model: UMiKTGATModel,
    records_by_learner: dict,
    config: dict,
    device: torch.device,
) -> dict:
    """
    Evaluate model on a dataset split. Returns dict of metrics.

    All metrics are computed from actual model output — no values are assumed.
    """
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error

    model.eval()
    seq_len = config.get("seq_len", 10)

    all_mastery_pred = []
    all_mastery_gt = []
    all_mc_pred = []
    all_mc_gt = []
    all_ret_pred = []
    all_ret_gt = []

    for lid, recs in records_by_learner.items():
        if not recs:
            continue
        x_seq, labels = build_sequences(recs, seq_len=seq_len, device=device)

        prior_m = torch.full((N_CONCEPTS,), 0.3, device=device)
        prior_r = torch.full((N_CONCEPTS,), 0.8, device=device)
        learner_states = torch.stack([
            prior_m,
            torch.zeros(N_CONCEPTS, device=device),
            torch.full((N_CONCEPTS,), 0.5, device=device),
            prior_r,
        ], dim=1)

        outputs = model(x_seq, learner_states)
        mask = labels["active_mask"].cpu().numpy()
        if not mask.any():
            continue

        all_mastery_pred.extend(outputs["mastery"].cpu().numpy()[mask].tolist())
        all_mastery_gt.extend(labels["mastery"].cpu().numpy()[mask].tolist())

        if "misconception" in outputs:
            mc_probs = torch.softmax(outputs["misconception"], dim=-1)
            mc_pred_class = mc_probs.argmax(dim=-1).cpu().numpy()[mask]
            all_mc_pred.extend(mc_pred_class.tolist())
            all_mc_gt.extend(labels["misconception"].cpu().numpy()[mask].tolist())

        if "retention" in outputs:
            all_ret_pred.extend(outputs["retention"].cpu().numpy()[mask].tolist())
            all_ret_gt.extend(labels["retention"].cpu().numpy()[mask].tolist())

    metrics = {}

    # ── Mastery metrics ───────────────────────────────────────────────────────
    if all_mastery_pred and all_mastery_gt:
        pred_np = np.array(all_mastery_pred)
        gt_np = np.array(all_mastery_gt)

        # Binary accuracy using 0.5 threshold, or median fallback if single class
        thresh = 0.5 if len(np.unique((gt_np >= 0.5).astype(int))) > 1 else float(np.median(gt_np))
        pred_binary = (pred_np >= thresh).astype(int)
        gt_binary = (gt_np >= thresh).astype(int)
        metrics["mastery_accuracy"] = float(accuracy_score(gt_binary, pred_binary))
        metrics["mastery_rmse"] = float(np.sqrt(mean_squared_error(gt_np, pred_np)))

        try:
            if len(np.unique(gt_binary)) > 1:
                metrics["mastery_auc"] = float(roc_auc_score(gt_binary, pred_np))
            else:
                metrics["mastery_auc"] = float("nan")
        except Exception:
            metrics["mastery_auc"] = float("nan")

    # ── Misconception metrics ─────────────────────────────────────────────────
    if all_mc_pred and all_mc_gt:
        try:
            metrics["mc_f1_macro"] = float(f1_score(
                all_mc_gt, all_mc_pred, average="macro", zero_division=0
            ))
            metrics["mc_accuracy"] = float(accuracy_score(all_mc_gt, all_mc_pred))
        except Exception:
            metrics["mc_f1_macro"] = float("nan")
            metrics["mc_accuracy"] = float("nan")

    # ── Retention metrics ─────────────────────────────────────────────────────
    if all_ret_pred and all_ret_gt:
        pred_r = np.array(all_ret_pred)
        gt_r = np.array(all_ret_gt)
        metrics["retention_rmse"] = float(np.sqrt(mean_squared_error(gt_r, pred_r)))
        metrics["retention_mae"] = float(np.mean(np.abs(pred_r - gt_r)))

    return metrics


def train(config: dict, variant_name: str = "default") -> dict:
    """
    Full training loop. Returns final evaluation metrics on test set.

    All metrics come from actual execution — no fabricated values.
    """
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data (synthetic — labeled as such) ───────────────────────────────
    print("\nLoading (synthetic) training data...")
    train_recs = load_synthetic_split("train")
    val_recs = load_synthetic_split("val")
    test_recs = load_synthetic_split("test")
    print(f"  Train: {len(train_recs)} interactions | "
          f"Val: {len(val_recs)} | Test: {len(test_recs)}")

    train_by_learner = group_by_learner(train_recs)
    val_by_learner = group_by_learner(val_recs)
    test_by_learner = group_by_learner(test_recs)

    # ── Build model ───────────────────────────────────────────────────────────
    model = UMiKTGATModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-5
    )

    # ── Loss functions ────────────────────────────────────────────────────────
    mastery_loss_fn = nn.BCELoss()

    mc_loss_fn = None
    if config.get("use_misconception_head", True):
        class_weights = compute_class_weights(train_recs).to(device)
        mc_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    ret_loss_fn = None
    if config.get("use_retention", True):
        ret_loss_fn = nn.MSELoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    n_epochs = config.get("n_epochs", 50)
    patience = config.get("patience", 10)
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    train_log = []

    results_dir = (
        Path(__file__).parent.parent
        / "experiments" / "results" / variant_name
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = results_dir / "best_model.pt"

    print(f"\nTraining '{variant_name}' for up to {n_epochs} epochs...")
    t0 = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        train_losses = train_epoch(
            model, train_by_learner, optimizer, config, device,
            mastery_loss_fn, mc_loss_fn, ret_loss_fn,
        )
        val_metrics = evaluate(model, val_by_learner, config, device)

        val_mastery_loss = val_metrics.get("mastery_rmse", 1.0)
        scheduler.step(val_mastery_loss)

        log_entry = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_losses.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        train_log.append(log_entry)

        if val_mastery_loss < best_val_loss:
            best_val_loss = val_mastery_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{n_epochs} | "
                f"train_loss={train_losses['total']:.4f} | "
                f"val_mastery_acc={val_metrics.get('mastery_accuracy', 0):.4f} | "
                f"val_auc={val_metrics.get('mastery_auc', 0):.4f}"
            )

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (best: {best_epoch})")
            break

    elapsed = time.perf_counter() - t0
    print(f"  Training time: {elapsed:.1f}s")

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\nLoading best checkpoint for test evaluation...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_metrics = evaluate(model, test_by_learner, config, device)

    print(f"  Test Mastery Accuracy:  {test_metrics.get('mastery_accuracy', 0):.4f}")
    print(f"  Test Mastery AUC:       {test_metrics.get('mastery_auc', 0):.4f}")
    print(f"  Test Mastery RMSE:      {test_metrics.get('mastery_rmse', 0):.4f}")
    if "mc_f1_macro" in test_metrics:
        print(f"  Test MC F1 (macro):     {test_metrics.get('mc_f1_macro', 0):.4f}")
    if "retention_rmse" in test_metrics:
        print(f"  Test Retention RMSE:    {test_metrics.get('retention_rmse', 0):.4f}")

    # ── Save loss curve plot ──────────────────────────────────────────────────
    epochs_x = [log["epoch"] for log in train_log]
    train_losses_y = [log["train_total"] for log in train_log]
    val_rmse_y = [log.get("val_mastery_rmse", 0) for log in train_log]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.plot(epochs_x, train_losses_y, label="Train Total Loss", color="#1f77b4", linewidth=2)
    plt.plot(epochs_x, val_rmse_y, label="Val Mastery RMSE", color="#ff7f0e", linestyle="--", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Metric")
    plt.title(f"Training Loss Curve — {variant_name} (synthetic data)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "train_loss_curve.png", dpi=150)
    plt.close()

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "variant": variant_name,
        "config": config,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "training_time_sec": round(elapsed, 2),
        "test_metrics": test_metrics,
        "data_note": "All data is SYNTHETIC. See DATASET_CARD.md.",
    }

    with open(results_dir / "train_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(results_dir / "train_log.json", "w") as f:
        json.dump(train_log, f, indent=2)

    print(f"\nLoss curve saved: {results_dir / 'train_loss_curve.png'}")
    print(f"Results saved to:  {results_dir}")
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UMiKT-GAT model")
    parser.add_argument("--variant", type=str, default="full_umikt_gat",
                        help="Ablation variant name")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    args = parser.parse_args()

    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        variant_name = Path(args.config).stem
    else:
        config = make_ablation_config(args.variant)
        variant_name = args.variant

    train(config, variant_name=variant_name)
