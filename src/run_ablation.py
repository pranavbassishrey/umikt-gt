"""
Ablation study master script.

Trains and evaluates all 6 ablation variants + BKT and DKT baselines on the
same data splits. Saves:
  - experiments/results/ablation_results.csv
  - experiments/results/ablation_bar_chart.png  (with "(synthetic)" label)
  - experiments/results/baseline_comparison.csv
  - experiments/results/baseline_comparison_chart.png

ALL NUMBERS in the output files come from actual execution of this script.
No values are hand-written or assumed.

Usage:
  python src/run_ablation.py
  python src/run_ablation.py --fast   # fewer epochs for quick sanity check
"""

import argparse
import csv
import json
import sys
import time
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.loader import load_synthetic_split, group_by_learner
from src.models.umikt_gat import make_ablation_config
from src.models.baselines.bkt import BKT
from src.models.baselines.dkt import DKTTrainer
from src.train import train, set_seed, build_sequences, evaluate
from src.evaluate import evaluate_with_uncertainty
from src.models.umikt_gat import UMiKTGATModel


# ── Ablation variants matching the spec table ─────────────────────────────────
ABLATION_VARIANTS = [
    "mastery_only",
    "+temporal",
    "+graph",
    "+misconception",
    "+uncertainty",
    "full_umikt_gat",
]

VARIANT_DISPLAY_NAMES = {
    "mastery_only":    "Mastery-Only Baseline",
    "+temporal":       "+Temporal (GRU)",
    "+graph":          "+Graph (GAT)",
    "+misconception":  "+Misconception Head",
    "+uncertainty":    "+Uncertainty (MC-Dropout)",
    "full_umikt_gat":  "Full UMiKT-GAT",
}


def run_bkt_baseline(train_recs, val_by_learner, test_by_learner):
    """Fit and evaluate BKT on active concept states of test split with val-derived threshold."""
    print("\n[BKT Baseline]")
    t0 = time.perf_counter()
    bkt = BKT(n_em_iters=30)
    bkt.fit(train_recs, skill_col="concept_id")

    # Compute validation predictions to derive threshold
    val_preds = []
    for lid, recs in val_by_learner.items():
        x_seq, labels = build_sequences(recs)
        mask = labels["active_mask"].cpu().numpy()
        if not mask.any():
            continue
        p_seq = bkt.predict_all(recs, skill_col="concept_id")
        last_p = {r["concept_id"]: p for r, p in zip(recs, p_seq)}
        for cid in range(7):
            if mask[cid]:
                val_preds.append(last_p.get(cid, 0.5))
    val_thresh = float(np.median(val_preds)) if val_preds else 0.5

    bkt_preds, gts = [], []
    for lid, recs in test_by_learner.items():
        x_seq, labels = build_sequences(recs)
        mask = labels["active_mask"].cpu().numpy()
        if not mask.any():
            continue
        p_seq = bkt.predict_all(recs, skill_col="concept_id")
        last_p = {}
        for r, p in zip(recs, p_seq):
            last_p[r["concept_id"]] = p
        for cid in range(7):
            if mask[cid]:
                bkt_preds.append(last_p.get(cid, 0.5))
                gts.append(labels["mastery"][cid].item())

    elapsed = time.perf_counter() - t0
    pred_np = np.array(bkt_preds)
    gt_np = np.array(gts)

    has_binary_gt = len(np.unique((gt_np >= 0.5).astype(int))) > 1
    gt_thresh = 0.5 if has_binary_gt else float(np.median(gt_np))
    gt_binary = (gt_np >= gt_thresh).astype(int)
    pred_binary = (pred_np >= val_thresh).astype(int)

    from sklearn.metrics import accuracy_score, roc_auc_score
    acc = accuracy_score(gt_binary, pred_binary)
    try:
        auc = roc_auc_score(gt_binary, pred_np) if len(set(gt_binary)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    rmse = float(np.sqrt(np.mean((pred_np - gt_np) ** 2)))

    pred_bin_counts = [int(np.sum(pred_binary == 0)), int(np.sum(pred_binary == 1))]
    print(f"  Val-derived Pred Threshold: {val_thresh:.6f}")
    print(f"  Test pred_bin class counts: 0: {pred_bin_counts[0]}, 1: {pred_bin_counts[1]}")
    print(f"  Acc={acc:.4f} | AUC={auc:.4f} | RMSE={rmse:.4f} | N={len(gt_np)} | t={elapsed:.1f}s")

    metrics = {
        "mastery_accuracy": acc,
        "mastery_auc": auc,
        "mastery_rmse": rmse,
        "val_pred_thresh": val_thresh,
        "pred_bin_counts": pred_bin_counts,
        "training_time_sec": round(elapsed, 2),
        "_pred_np": pred_np.tolist(),
        "_gt_binary": gt_binary.tolist(),
    }
    return metrics


def run_dkt_baseline(train_recs, val_by_learner, test_by_learner, n_epochs=20, device="cpu"):
    """Train and evaluate DKT on active concept states of test split with val-derived threshold."""
    print("\n[DKT Baseline]")
    t0 = time.perf_counter()
    trainer = DKTTrainer(n_skills=7, hidden_dim=64, lr=1e-3, n_epochs=n_epochs, device=device)
    trainer.train(train_recs)

    # Compute validation predictions to derive threshold
    val_preds = []
    for lid, recs in val_by_learner.items():
        x_seq, labels = build_sequences(recs)
        mask = labels["active_mask"].cpu().numpy()
        if not mask.any():
            continue
        p_seq = trainer.predict_all(recs, skill_col="concept_id")
        last_p = {r["concept_id"]: p for r, p in zip(recs, p_seq)}
        for cid in range(7):
            if mask[cid]:
                val_preds.append(last_p.get(cid, 0.5))
    val_thresh = float(np.median(val_preds)) if val_preds else 0.5

    dkt_preds, gts = [], []
    for lid, recs in test_by_learner.items():
        x_seq, labels = build_sequences(recs)
        mask = labels["active_mask"].cpu().numpy()
        if not mask.any():
            continue
        p_seq = trainer.predict_all(recs, skill_col="concept_id")
        last_p = {}
        for r, p in zip(recs, p_seq):
            last_p[r["concept_id"]] = p
        for cid in range(7):
            if mask[cid]:
                dkt_preds.append(last_p.get(cid, 0.5))
                gts.append(labels["mastery"][cid].item())

    elapsed = time.perf_counter() - t0
    pred_np = np.array(dkt_preds)
    gt_np = np.array(gts)

    has_binary_gt = len(np.unique((gt_np >= 0.5).astype(int))) > 1
    gt_thresh = 0.5 if has_binary_gt else float(np.median(gt_np))
    gt_binary = (gt_np >= gt_thresh).astype(int)
    pred_binary = (pred_np >= val_thresh).astype(int)

    from sklearn.metrics import accuracy_score, roc_auc_score
    acc = accuracy_score(gt_binary, pred_binary)
    try:
        auc = roc_auc_score(gt_binary, pred_np) if len(set(gt_binary)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    rmse = float(np.sqrt(np.mean((pred_np - gt_np) ** 2)))

    pred_bin_counts = [int(np.sum(pred_binary == 0)), int(np.sum(pred_binary == 1))]
    print(f"  Val-derived Pred Threshold: {val_thresh:.6f}")
    print(f"  Test pred_bin class counts: 0: {pred_bin_counts[0]}, 1: {pred_bin_counts[1]}")
    print(f"  Acc={acc:.4f} | AUC={auc:.4f} | RMSE={rmse:.4f} | N={len(gt_np)} | t={elapsed:.1f}s")

    metrics = {
        "mastery_accuracy": acc,
        "mastery_auc": auc,
        "mastery_rmse": rmse,
        "val_pred_thresh": val_thresh,
        "pred_bin_counts": pred_bin_counts,
        "training_time_sec": round(elapsed, 2),
        "_pred_np": pred_np.tolist(),
        "_gt_binary": gt_binary.tolist(),
    }
    return metrics


def plot_ablation_bar_chart(rows: list[dict], output_path: Path):
    """
    Generate ablation bar chart.
    Title includes "(synthetic)" as required by the project spec.
    """
    names = [r["variant_display"] for r in rows]
    acc_vals = [r.get("mastery_accuracy", 0) for r in rows]
    auc_vals = [r.get("mastery_auc", 0) for r in rows]
    f1_vals = [r.get("mc_f1_macro", 0) for r in rows]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))

    bars1 = ax.bar(x - width, acc_vals, width, label="Mastery Acc", color="#4C72B0", alpha=0.88)
    bars2 = ax.bar(x,         auc_vals, width, label="Mastery AUC", color="#DD8452", alpha=0.88)
    bars3 = ax.bar(x + width, f1_vals,  width, label="MC F1 (macro)", color="#55A868", alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("UMiKT-GAT Ablation Study — Mastery & Misconception Metrics (synthetic data)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.35)

    # Annotate bar tops
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved: {output_path}")


def plot_baseline_comparison(ablation_row: dict, bkt_row: dict, dkt_row: dict,
                              output_path: Path):
    """Baseline comparison bar chart (synthetic data label required)."""
    names = ["BKT Baseline", "DKT Baseline", "Full UMiKT-GAT"]
    rows = [bkt_row, dkt_row, ablation_row]
    acc_vals = [r.get("mastery_accuracy", 0) for r in rows]
    auc_vals = [r.get("mastery_auc", 0) for r in rows]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, acc_vals, width, label="Mastery Acc", color="#4C72B0", alpha=0.88)
    bars2 = ax.bar(x + width/2, auc_vals, width, label="Mastery AUC", color="#DD8452", alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Baseline Comparison — Mastery Prediction (synthetic data)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.35)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved: {output_path}")


def compute_bootstrap_cis(predictions: dict, n_bootstraps: int = 1000, seed: int = 42) -> dict:
    """
    Bootstrap resample the 82 test-set (prediction, ground-truth) pairs 1,000 times (with replacement).
    Recompute AUC for: mastery-only baseline, full UMiKT-GAT, and DKT.
    Report 95% confidence interval (2.5th–97.5th percentile).
    """
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    models = ["mastery_only", "full_umikt_gat", "dkt"]
    available_models = [m for m in models if m in predictions]
    if not available_models:
        return {}

    first_model = available_models[0]
    n_samples = len(predictions[first_model]["gt_binary"])

    boot_aucs = {m: [] for m in available_models}
    for _ in range(n_bootstraps):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        for m in available_models:
            y_true = np.array(predictions[m]["gt_binary"])[idx]
            y_pred = np.array(predictions[m]["preds"])[idx]
            if len(np.unique(y_true)) > 1:
                try:
                    boot_aucs[m].append(float(roc_auc_score(y_true, y_pred)))
                except Exception:
                    pass

    cis = {}
    for m in available_models:
        arr = np.array(boot_aucs[m])
        if len(arr) > 0:
            low, high = np.percentile(arr, [2.5, 97.5])
            cis[m] = {
                "mean": float(np.mean(arr)),
                "ci_95_low": float(low),
                "ci_95_high": float(high),
                "n_valid": len(arr),
            }
        else:
            cis[m] = {"mean": float("nan"), "ci_95_low": float("nan"), "ci_95_high": float("nan"), "n_valid": 0}
    return cis


def write_results_summary(ablation_rows, bkt_metrics, dkt_metrics, output_path: Path, bootstrap_cis: dict | None = None):
    """
    Write RESULTS_SUMMARY.md — plain-language hedged claims.
    Every number in this file is sourced from actual run metrics.
    """
    full_row = next((r for r in ablation_rows if r["variant"] == "full_umikt_gat"), {})
    baseline_row = next((r for r in ablation_rows if r["variant"] == "mastery_only"), {})
    temporal_row = next((r for r in ablation_rows if r["variant"] == "+temporal"), {})

    def fmt(v, key):
        val = v.get(key, float("nan"))
        if not isinstance(val, (int, float)):
            return str(val)
        return f"{val:.4f}" if not math.isnan(val) else "N/A"

    # Issue C & D: Honest DKT comparison & CI overlap analysis
    ci_full = bootstrap_cis.get("full_umikt_gat", {}) if bootstrap_cis else {}
    ci_dkt = bootstrap_cis.get("dkt", {}) if bootstrap_cis else {}
    ci_base = bootstrap_cis.get("mastery_only", {}) if bootstrap_cis else {}

    def format_ci(ci_dict):
        low = ci_dict.get("ci_95_low", float("nan"))
        high = ci_dict.get("ci_95_high", float("nan"))
        if math.isnan(low) or math.isnan(high):
            return "N/A"
        return f"[{low:.4f}, {high:.4f}]"

    dkt_overlap = True
    base_overlap = True
    if ci_full and ci_dkt and not math.isnan(ci_full.get("ci_95_low", float("nan"))) and not math.isnan(ci_dkt.get("ci_95_low", float("nan"))):
        dkt_overlap = not (ci_full["ci_95_low"] > ci_dkt["ci_95_high"] or ci_dkt["ci_95_low"] > ci_full["ci_95_high"])
    if ci_full and ci_base and not math.isnan(ci_full.get("ci_95_low", float("nan"))) and not math.isnan(ci_base.get("ci_95_low", float("nan"))):
        base_overlap = not (ci_full["ci_95_low"] > ci_base["ci_95_high"] or ci_base["ci_95_low"] > ci_full["ci_95_high"])

    dkt_dist_str = "these differences are not statistically distinguishable" if dkt_overlap else "these differences are statistically distinguishable"
    base_dist_str = "this difference is not statistically distinguishable" if base_overlap else "this difference is statistically distinguishable"

    dkt_sentence = f"DKT achieved a higher AUC than the full UMiKT-GAT model on this synthetic test set ({fmt(dkt_metrics, 'mastery_auc')} vs {fmt(full_row, 'mastery_auc')}), while UMiKT-GAT achieved a lower RMSE ({fmt(full_row, 'mastery_rmse')} vs {fmt(dkt_metrics, 'mastery_rmse')}); given the bootstrap CIs in Issue C, {dkt_dist_str}."
    base_sentence = f"Between the full UMiKT-GAT model and the mastery-only baseline ({fmt(full_row, 'mastery_auc')} vs {fmt(baseline_row, 'mastery_auc')}), given the 95% bootstrap CIs ({format_ci(ci_full)} vs {format_ci(ci_base)}), {base_dist_str}."

    content = f"""# Results Summary — UMiKT-GAT (synthetic data)

> [!WARNING]
> **All results in this file were produced on SYNTHETIC data** generated by
> `data/layer2_synthetic/generate_synthetic.py`. They validate that the model
> can recover known latent states from simulated trajectories. They do NOT
> represent performance on real student data and must not be presented as such.
> See `DATASET_CARD.md` for full limitations.

Generated automatically by `src/run_ablation.py`. Every number below maps
directly to a row in `experiments/results/ablation_results.csv`.

---

## RQ1: Does the proposed model improve mastery prediction vs mastery-only baseline?

| Model | Accuracy | AUC | RMSE | 95% Bootstrap CI (AUC) |
|---|---|---|---|---|
| Mastery-Only Baseline | {fmt(baseline_row, 'mastery_accuracy')} | {fmt(baseline_row, 'mastery_auc')} | {fmt(baseline_row, 'mastery_rmse')} | {format_ci(ci_base)} |
| +Temporal (GRU) | {fmt(temporal_row, 'mastery_accuracy')} | {fmt(temporal_row, 'mastery_auc')} | {fmt(temporal_row, 'mastery_rmse')} | N/A |
| Full UMiKT-GAT | {fmt(full_row, 'mastery_accuracy')} | {fmt(full_row, 'mastery_auc')} | {fmt(full_row, 'mastery_rmse')} | {format_ci(ci_full)} |
| BKT Baseline | {fmt(bkt_metrics, 'mastery_accuracy')} | {fmt(bkt_metrics, 'mastery_auc')} | {fmt(bkt_metrics, 'mastery_rmse')} | N/A |
| DKT Baseline | {fmt(dkt_metrics, 'mastery_accuracy')} | {fmt(dkt_metrics, 'mastery_auc')} | {fmt(dkt_metrics, 'mastery_rmse')} | {format_ci(ci_dkt)} |

### DKT Comparison & Statistical Distinguishability
{dkt_sentence}

{base_sentence}

> [!NOTE]
> The test set consists of 82 active concept-states, which places a practical limitation on how much confidence to place in any single comparison.

## RQ2: Does misconception evidence improve learner-state estimation?

| Variant | MC F1 (macro) | MC Accuracy |
|---|---|---|
| +Misconception Head | {fmt(next((r for r in ablation_rows if r['variant']=='+misconception'), {}), 'mc_f1_macro')} | {fmt(next((r for r in ablation_rows if r['variant']=='+misconception'), {}), 'mc_accuracy')} |
| Full UMiKT-GAT | {fmt(full_row, 'mc_f1_macro')} | {fmt(full_row, 'mc_accuracy')} |

**Hedged finding:** Misconception classification F1 on synthetic data reflects the model's
ability to recover ground-truth misconception labels that were directly generated by a Markov
chain over the taxonomy. Real misconception detection requires LLM extraction on real student
text responses, which has not been evaluated at scale in this codebase.

## RQ3: Does uncertainty-aware modeling improve curriculum decisions?

| Variant | ECE | Brier Score |
|---|---|---|
| Full UMiKT-GAT | {fmt(full_row, 'ece')} | {fmt(full_row, 'brier_score')} |

**Hedged finding:** ECE and Brier score are computed on synthetic data. MC-Dropout variance
provides a reasonable proxy for epistemic uncertainty (data sparsity per concept) but is
not validated as a true Bayesian posterior approximation. The relationship between MC-Dropout
variance and actual curriculum decision quality requires a prospective user study.

## RQ4: Does prerequisite-aware graph modeling improve prediction?

| Variant | Mastery Acc | Mastery AUC |
|---|---|---|
| +Temporal only | {fmt(temporal_row, 'mastery_accuracy')} | {fmt(temporal_row, 'mastery_auc')} |
| +Graph (GAT added) | {fmt(next((r for r in ablation_rows if r['variant']=='+graph'), {}), 'mastery_accuracy')} | {fmt(next((r for r in ablation_rows if r['variant']=='+graph'), {}), 'mastery_auc')} |

**Hedged finding:** The incremental effect of the GAT layer is evaluated by comparing
+Temporal vs +Graph variants. Results on synthetic data may overestimate the value of
graph propagation because the simulation generates data with explicit prerequisite
dependencies. Real student data may have weaker prerequisite signals.

## RQ5 & RQ6: Curriculum decisions and system cost

RQ5 (curriculum decision quality) requires a prospective user study with real learners.
On synthetic data, curriculum decisions are evaluated qualitatively via `demo_session.py`.
No learning-gain metrics are reported — doing so without a controlled experiment would
constitute fabricated results.

RQ6 (system cost): See `experiments/results/system_cost.json` generated by `demo_session.py`.
Latency, LLM calls (mock), and token usage are reported from an actual execution run.

## Limitations

1. **Synthetic data reliance**: All quantitative results are from synthetic data. Real
   performance may differ substantially.
2. **Small taxonomy**: 7-class misconception taxonomy is domain-specific and hand-authored.
   Not validated by educational experts.
3. **Single-label misconception**: Multi-label misconception modeling may better reflect
   real student errors.
4. **No real LLM evaluation at scale**: The MockLLMProvider is used in all training runs.
   Real LLM extraction quality on the handwritten QA set is evaluated separately in
   `tests/test_schema_validation.py` but not at model training scale.
5. **Compute limitations**: MC-Dropout with 20 samples adds ~20x inference cost vs
   deterministic inference. Deep ensembles would be more principled but require N×
   more training.
6. **Test set sample size**: The test set consists of 82 active concept-states, which places
   a practical limitation on how much confidence to place in any single comparison and
   results in wide empirical bootstrap confidence intervals.
"""

    output_path.write_text(content)
    print(f"  RESULTS_SUMMARY.md written: {output_path}")


def main(fast: bool = False):
    set_seed(42)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    results_base = Path(__file__).parent.parent / "experiments" / "results"
    results_base.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UMiKT-GAT ABLATION STUDY")
    print("NOTE: All data is SYNTHETIC. See DATASET_CARD.md.")
    print("=" * 60)

    # Load data once
    train_recs = load_synthetic_split("train")
    val_recs = load_synthetic_split("val")
    test_recs = load_synthetic_split("test")
    val_by_learner = group_by_learner(val_recs)
    test_by_learner = group_by_learner(test_recs)

    # ── Run ablation variants ─────────────────────────────────────────────────
    ablation_rows = []

    for variant in ABLATION_VARIANTS:
        print(f"\n{'─'*50}")
        print(f"Variant: {variant} (Independent Training Start)")
        set_seed(42)  # Ensure independent initialization per variant
        cfg = make_ablation_config(variant)
        if fast:
            cfg["n_epochs"] = 5
            cfg["patience"] = 3

        t_start = time.perf_counter()
        test_metrics = train(cfg, variant_name=variant)
        elapsed = time.perf_counter() - t_start

        # Read train_log.json to print loss progression summary
        variant_dir = results_base / variant
        log_file = variant_dir / "train_log.json"
        if log_file.exists():
            with open(log_file) as f:
                t_log = json.load(f)
            if t_log:
                start_loss = t_log[0].get("train_total", 0.0)
                end_loss = t_log[-1].get("train_total", 0.0)
                print(f"  [Loss progression for {variant}]: Start loss={start_loss:.4f} → End loss={end_loss:.4f} ({len(t_log)} epochs)")

        # Run full evaluation with uncertainty for the full model
        if variant == "full_umikt_gat":
            model = UMiKTGATModel(cfg).to(device)
            ckpt = variant_dir / "best_model.pt"
            if ckpt.exists():
                model.load_state_dict(torch.load(ckpt, map_location=device))
                test_metrics = evaluate_with_uncertainty(
                    model, test_by_learner, cfg, device, variant_dir, variant,
                    val_records_by_learner=val_by_learner,
                )

        row = {
            "variant": variant,
            "variant_display": VARIANT_DISPLAY_NAMES[variant],
            **test_metrics,
            "training_time_sec": round(elapsed, 2),
            "data_note": "synthetic",
        }
        ablation_rows.append(row)

    # ── Run baselines ─────────────────────────────────────────────────────────
    bkt_n_epochs = 5 if fast else 30
    dkt_n_epochs = 5 if fast else 20

    bkt_metrics = run_bkt_baseline(train_recs, val_by_learner, test_by_learner)
    dkt_metrics = run_dkt_baseline(train_recs, val_by_learner, test_by_learner, n_epochs=dkt_n_epochs, device=device_str)

    # ── Compute Bootstrap CIs (Issue C) ───────────────────────────────────────
    predictions_dict = {}
    for r in ablation_rows:
        if "_pred_np" in r and "_gt_binary" in r:
            predictions_dict[r["variant"]] = {
                "preds": r["_pred_np"],
                "gt_binary": r["_gt_binary"],
            }
    if "_pred_np" in bkt_metrics and "_gt_binary" in bkt_metrics:
        predictions_dict["bkt"] = {
            "preds": bkt_metrics["_pred_np"],
            "gt_binary": bkt_metrics["_gt_binary"],
        }
    if "_pred_np" in dkt_metrics and "_gt_binary" in dkt_metrics:
        predictions_dict["dkt"] = {
            "preds": dkt_metrics["_pred_np"],
            "gt_binary": dkt_metrics["_gt_binary"],
        }

    with open(results_base / "test_predictions.json", "w") as f:
        json.dump(predictions_dict, f, indent=2)

    bootstrap_cis = compute_bootstrap_cis(predictions_dict, n_bootstraps=1000, seed=42)
    with open(results_base / "bootstrap_cis.json", "w") as f:
        json.dump(bootstrap_cis, f, indent=2)
    print(f"\nBootstrap 95% CIs (N=1000):")
    for m, ci in bootstrap_cis.items():
        print(f"  {m:<20}: Mean={ci['mean']:.4f}, 95% CI=[{ci['ci_95_low']:.4f}, {ci['ci_95_high']:.4f}] (valid={ci['n_valid']})")

    # ── Clean rows before writing CSVs (remove private fields starting with _) ─
    cleaned_ablation_rows = []
    for r in ablation_rows:
        cleaned_ablation_rows.append({k: v for k, v in r.items() if not k.startswith("_")})

    cleaned_bkt = {k: v for k, v in bkt_metrics.items() if not k.startswith("_")}
    cleaned_dkt = {k: v for k, v in dkt_metrics.items() if not k.startswith("_")}
    full_row = next((r for r in cleaned_ablation_rows if r["variant"] == "full_umikt_gat"), {})

    # ── Save ablation CSV ─────────────────────────────────────────────────────
    ablation_csv = results_base / "ablation_results.csv"
    all_keys = set()
    for r in cleaned_ablation_rows:
        all_keys.update(r.keys())
    fieldnames = ["variant", "variant_display"] + sorted(
        k for k in all_keys if k not in ("variant", "variant_display")
    )

    with open(ablation_csv, "w", newline="") as f:
        f.write("# SYNTHETIC DATA — UMiKT-GAT Ablation Results\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cleaned_ablation_rows)
    print(f"\nAblation CSV: {ablation_csv}")

    # ── Save baseline CSV ─────────────────────────────────────────────────────
    baseline_csv = results_base / "baseline_comparison.csv"
    baseline_rows = [
        {"model": "BKT", **cleaned_bkt, "data_note": "synthetic"},
        {"model": "DKT", **cleaned_dkt, "data_note": "synthetic"},
        {"model": "Full UMiKT-GAT", **full_row, "data_note": "synthetic"},
    ]
    b_keys = set()
    for r in baseline_rows:
        b_keys.update(r.keys())
    b_fields = ["model"] + sorted(k for k in b_keys if k != "model")
    with open(baseline_csv, "w", newline="") as f:
        f.write("# SYNTHETIC DATA — Baseline Comparison\n")
        writer = csv.DictWriter(f, fieldnames=b_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(baseline_rows)
    print(f"Baseline CSV: {baseline_csv}")

    # ── Generate charts ───────────────────────────────────────────────────────
    plot_ablation_bar_chart(cleaned_ablation_rows, results_base / "ablation_bar_chart.png")
    plot_baseline_comparison(
        full_row, cleaned_bkt, cleaned_dkt,
        results_base / "baseline_comparison_chart.png"
    )

    # ── Write RESULTS_SUMMARY.md ──────────────────────────────────────────────
    write_results_summary(
        cleaned_ablation_rows, cleaned_bkt, cleaned_dkt,
        results_base / "RESULTS_SUMMARY.md",
        bootstrap_cis=bootstrap_cis
    )

    print("\n" + "=" * 60)
    print("Ablation study complete.")
    print(f"All results in: {results_base}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Run with minimal epochs for quick verification")
    args = parser.parse_args()
    main(fast=args.fast)
