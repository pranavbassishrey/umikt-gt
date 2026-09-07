"""
Evaluation script for UMiKT-GAT.

Computes:
  - ECE (Expected Calibration Error) for uncertainty calibration
  - Brier Score for probabilistic prediction quality
  - Reliability diagram (saved as PNG)
  - All metrics saved to experiments/results/{variant}/eval_metrics.json

All numbers come from actual model execution on the test split.
No values are assumed or fabricated.

Usage:
  python src/evaluate.py --variant full_umikt_gat
"""

import sys
import json
import argparse
import math
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.loader import load_synthetic_split, group_by_learner
from src.features.feature_builder import N_CONCEPTS, INPUT_DIM
from src.models.umikt_gat import UMiKTGATModel, make_ablation_config
from src.models.mc_dropout_wrapper import MCDropoutWrapper
from src.train import build_sequences


def compute_ece(
    pred_probs: np.ndarray,
    true_labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures how well predicted probabilities match empirical frequencies.
    A well-calibrated model has ECE close to 0.

    Formula: ECE = Σ (|B_m| / N) * |acc(B_m) - conf(B_m)|
    Where B_m is the set of predictions in the m-th confidence bin.

    Args:
        pred_probs: Predicted probabilities in [0,1]. Shape: (N,)
        true_labels: Binary true labels {0,1}. Shape: (N,)
        n_bins: Number of equal-width confidence bins (default 15, standard).

    Returns:
        ECE value ∈ [0,1]. Lower is better.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(pred_probs)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (pred_probs >= lo) & (pred_probs < hi)
        if mask.sum() == 0:
            continue
        bin_preds = pred_probs[mask]
        bin_true = true_labels[mask]
        conf = bin_preds.mean()
        acc = bin_true.mean()
        ece += (mask.sum() / n) * abs(acc - conf)

    return float(ece)


def compute_brier_score(
    pred_probs: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """
    Compute Brier Score.

    Brier Score = (1/N) Σ (p_i - y_i)^2
    Lower is better (range [0,1] for binary). Perfect = 0, random = 0.25.

    Args:
        pred_probs: Predicted probabilities in [0,1].
        true_labels: Binary true labels {0,1}.

    Returns:
        Brier Score.
    """
    return float(np.mean((pred_probs - true_labels.astype(float)) ** 2))


def plot_reliability_diagram(
    pred_probs: np.ndarray,
    true_labels: np.ndarray,
    output_path: Path,
    title_suffix: str = "",
    n_bins: int = 15,
):
    """
    Save a reliability (calibration) diagram to a PNG file.

    A perfectly calibrated model would fall on the diagonal.
    Deviations show over- or under-confidence.

    Note: "(synthetic)" is included in the title as required by the project spec.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (pred_probs >= lo) & (pred_probs < hi)
        if mask.sum() == 0:
            continue
        bin_centers.append(pred_probs[mask].mean())
        bin_accuracies.append(true_labels[mask].mean())
        bin_counts.append(mask.sum())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Reliability diagram
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1.5)
    ax1.scatter(bin_centers, bin_accuracies, s=80, zorder=5, color="#4C72B0")
    ax1.plot(bin_centers, bin_accuracies, color="#4C72B0", linewidth=1.5, label="Model")
    ax1.set_xlabel("Mean Predicted Probability (Confidence)", fontsize=12)
    ax1.set_ylabel("Fraction of Positives (Accuracy)", fontsize=12)
    ax1.set_title(f"Reliability Diagram (synthetic){title_suffix}", fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Histogram of predicted probabilities
    ax2.hist(pred_probs, bins=20, color="#4C72B0", edgecolor="white", alpha=0.8)
    ax2.set_xlabel("Predicted Probability", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Distribution of Predicted Probabilities (synthetic)", fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Reliability diagram saved: {output_path}")


@torch.no_grad()
def evaluate_with_uncertainty(
    model: UMiKTGATModel,
    records_by_learner: dict,
    config: dict,
    device: torch.device,
    results_dir: Path,
    variant_name: str = "full_umikt_gat",
) -> dict:
    """
    Full evaluation including MC-Dropout uncertainty estimation.

    Returns dict of all computed metrics.
    """
    seq_len = config.get("seq_len", 10)
    mc_samples = config.get("mc_samples", 20)
    mc_wrapper = MCDropoutWrapper(model, n_samples=mc_samples, device=str(device))

    all_mastery_mean = []
    all_mastery_var = []
    all_mastery_gt_binary = []
    all_mastery_gt_cont = []
    all_mc_pred = []
    all_mc_gt = []
    all_ret_pred = []
    all_ret_gt = []

    for lid, recs in records_by_learner.items():
        if not recs:
            continue
        x_seq, labels = build_sequences(recs, seq_len=seq_len, device=device)
        mastery_gt = labels["mastery"]
        learner_states = torch.stack([
            mastery_gt,
            torch.zeros(N_CONCEPTS, device=device),
            torch.full((N_CONCEPTS,), 0.5, device=device),
            labels["retention"],
        ], dim=1)

        # MC-Dropout uncertainty
        def forward_fn(x_seq=x_seq, learner_states=learner_states):
            out = model(x_seq, learner_states)
            return out["mastery"]

        mean_pred, var_pred, _ = mc_wrapper.sample_predictions(forward_fn)

        all_mastery_mean.extend(mean_pred.numpy().tolist())
        all_mastery_var.extend(var_pred.numpy().tolist())
        all_mastery_gt_cont.extend(labels["mastery"].cpu().numpy().tolist())
        all_mastery_gt_binary.extend((labels["mastery"].cpu().numpy() >= 0.5).astype(int).tolist())

        # Deterministic pass for misconception / retention
        model.eval()
        outputs = model(x_seq, learner_states)

        if "misconception" in outputs:
            mc_pred = outputs["misconception"].argmax(dim=-1)
            all_mc_pred.extend(mc_pred.cpu().numpy().tolist())
            all_mc_gt.extend(labels["misconception"].cpu().numpy().tolist())

        if "retention" in outputs:
            all_ret_pred.extend(outputs["retention"].cpu().numpy().tolist())
            all_ret_gt.extend(labels["retention"].cpu().numpy().tolist())

    pred_np = np.array(all_mastery_mean)
    var_np = np.array(all_mastery_var)
    gt_binary = np.array(all_mastery_gt_binary)
    gt_cont = np.array(all_mastery_gt_cont)

    metrics = {}

    # Mastery metrics
    pred_binary = (pred_np >= 0.5).astype(int)
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, mean_squared_error
    metrics["mastery_accuracy"] = float(accuracy_score(gt_binary, pred_binary))
    metrics["mastery_rmse"] = float(np.sqrt(mean_squared_error(gt_cont, pred_np)))
    try:
        if len(np.unique(gt_binary)) > 1:
            metrics["mastery_auc"] = float(roc_auc_score(gt_binary, pred_np))
        else:
            metrics["mastery_auc"] = float("nan")
    except Exception:
        metrics["mastery_auc"] = float("nan")

    # Uncertainty metrics
    metrics["ece"] = compute_ece(pred_np, gt_binary)
    metrics["brier_score"] = compute_brier_score(pred_np, gt_binary)
    metrics["mean_predictive_variance"] = float(var_np.mean())

    # Misconception metrics
    if all_mc_pred and all_mc_gt:
        try:
            metrics["mc_f1_macro"] = float(f1_score(
                all_mc_gt, all_mc_pred, average="macro", zero_division=0
            ))
            metrics["mc_accuracy"] = float(accuracy_score(all_mc_gt, all_mc_pred))
        except Exception:
            metrics["mc_f1_macro"] = float("nan")
            metrics["mc_accuracy"] = float("nan")

    # Retention metrics
    if all_ret_pred and all_ret_gt:
        pred_r = np.array(all_ret_pred)
        gt_r = np.array(all_ret_gt)
        metrics["retention_rmse"] = float(np.sqrt(mean_squared_error(gt_r, pred_r)))
        metrics["retention_mae"] = float(np.mean(np.abs(pred_r - gt_r)))

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS — {variant_name} (synthetic data)")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:<30}: {v:.4f}" if not math.isnan(v) else f"  {k:<30}: nan")

    # ── Save reliability diagram ──────────────────────────────────────────────
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_reliability_diagram(
        pred_np, gt_binary,
        output_path=results_dir / "reliability_diagram.png",
        title_suffix=f" — {variant_name}",
    )

    # ── Save metrics ──────────────────────────────────────────────────────────
    eval_results = {
        "variant": variant_name,
        "metrics": metrics,
        "mc_samples": mc_samples,
        "data_note": "All data is SYNTHETIC. See DATASET_CARD.md.",
    }
    with open(results_dir / "eval_metrics.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\nEvaluation results saved to: {results_dir}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate UMiKT-GAT model")
    parser.add_argument("--variant", type=str, default="full_umikt_gat")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = make_ablation_config(args.variant)
    results_dir = (
        Path(__file__).parent.parent / "experiments" / "results" / args.variant
    )

    checkpoint = results_dir / "best_model.pt"
    if not checkpoint.exists():
        print(f"No checkpoint found at {checkpoint}. Run train.py first.")
        sys.exit(1)

    model = UMiKTGATModel(config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    test_recs = load_synthetic_split("test")
    test_by_learner = group_by_learner(test_recs)

    evaluate_with_uncertainty(model, test_by_learner, config, device, results_dir, args.variant)
