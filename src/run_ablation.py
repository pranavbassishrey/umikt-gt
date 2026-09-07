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
from src.train import train, set_seed
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


def run_bkt_baseline(train_recs, test_recs):
    """Fit and evaluate BKT on the test split."""
    print("\n[BKT Baseline]")
    t0 = time.perf_counter()
    bkt = BKT(n_em_iters=30)
    bkt.fit(train_recs, skill_col="concept_id")
    preds = bkt.predict_all(test_recs, skill_col="concept_id")
    elapsed = time.perf_counter() - t0

    gt = [float(r.get("gt_mastery", r.get("correctness", 0.0))) for r in test_recs]
    gt_binary = [1 if g >= 0.5 else 0 for g in gt]

    from sklearn.metrics import accuracy_score, roc_auc_score
    import numpy as np

    pred_np = np.array(preds)
    pred_binary = (pred_np >= 0.5).astype(int)
    acc = accuracy_score(gt_binary, pred_binary)
    try:
        auc = roc_auc_score(gt_binary, pred_np) if len(set(gt_binary)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    rmse = float(np.sqrt(np.mean((pred_np - np.array(gt)) ** 2)))

    metrics = {
        "mastery_accuracy": acc,
        "mastery_auc": auc,
        "mastery_rmse": rmse,
        "training_time_sec": round(elapsed, 2),
    }
    print(f"  Acc={acc:.4f} | AUC={auc:.4f} | RMSE={rmse:.4f} | t={elapsed:.1f}s")
    return metrics


def run_dkt_baseline(train_recs, test_recs, n_epochs=20, device="cpu"):
    """Train and evaluate DKT on the test split."""
    print("\n[DKT Baseline]")
    t0 = time.perf_counter()
    trainer = DKTTrainer(n_skills=7, hidden_dim=64, lr=1e-3, n_epochs=n_epochs, device=device)
    trainer.train(train_recs)
    preds = trainer.predict_all(test_recs, skill_col="concept_id")
    elapsed = time.perf_counter() - t0

    gt = [float(r.get("gt_mastery", r.get("correctness", 0.0))) for r in test_recs]
    gt_binary = [1 if g >= 0.5 else 0 for g in gt]

    from sklearn.metrics import accuracy_score, roc_auc_score
    import numpy as np

    pred_np = np.array(preds)
    pred_binary = (pred_np >= 0.5).astype(int)
    acc = accuracy_score(gt_binary, pred_binary)
    try:
        auc = roc_auc_score(gt_binary, pred_np) if len(set(gt_binary)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    rmse = float(np.sqrt(np.mean((pred_np - np.array(gt)) ** 2)))

    metrics = {
        "mastery_accuracy": acc,
        "mastery_auc": auc,
        "mastery_rmse": rmse,
        "training_time_sec": round(elapsed, 2),
    }
    print(f"  Acc={acc:.4f} | AUC={auc:.4f} | RMSE={rmse:.4f} | t={elapsed:.1f}s")
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


def write_results_summary(ablation_rows, bkt_metrics, dkt_metrics, output_path: Path):
    """
    Write RESULTS_SUMMARY.md — plain-language hedged claims.
    Every number in this file is sourced from actual run metrics.
    """
    full_row = next((r for r in ablation_rows if r["variant"] == "full_umikt_gat"), {})
    baseline_row = next((r for r in ablation_rows if r["variant"] == "mastery_only"), {})
    temporal_row = next((r for r in ablation_rows if r["variant"] == "+temporal"), {})

    def fmt(v, key):
        val = v.get(key, float("nan"))
        return f"{val:.4f}" if not math.isnan(val) else "N/A"

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

| Model | Accuracy | AUC | RMSE |
|---|---|---|---|
| Mastery-Only Baseline | {fmt(baseline_row, 'mastery_accuracy')} | {fmt(baseline_row, 'mastery_auc')} | {fmt(baseline_row, 'mastery_rmse')} |
| +Temporal (GRU) | {fmt(temporal_row, 'mastery_accuracy')} | {fmt(temporal_row, 'mastery_auc')} | {fmt(temporal_row, 'mastery_rmse')} |
| Full UMiKT-GAT | {fmt(full_row, 'mastery_accuracy')} | {fmt(full_row, 'mastery_auc')} | {fmt(full_row, 'mastery_rmse')} |
| BKT Baseline | {fmt(bkt_metrics, 'mastery_accuracy')} | {fmt(bkt_metrics, 'mastery_auc')} | {fmt(bkt_metrics, 'mastery_rmse')} |
| DKT Baseline | {fmt(dkt_metrics, 'mastery_accuracy')} | {fmt(dkt_metrics, 'mastery_auc')} | {fmt(dkt_metrics, 'mastery_rmse')} |

**Hedged finding:** On the synthetic dataset, we observe [improvement / no improvement /
inconclusive comparison] across ablation variants. Because the data is synthetic and the
ground-truth mastery trajectories were generated by a logistic growth model, the model that
most closely approximates that model's assumptions will benefit. Real-world generalization
requires evaluation on actual student interaction data.

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
    test_recs = load_synthetic_split("test")

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
                test_by_learner = group_by_learner(test_recs)
                test_metrics = evaluate_with_uncertainty(
                    model, test_by_learner, cfg, device, variant_dir, variant
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

    bkt_metrics = run_bkt_baseline(train_recs, test_recs)
    dkt_metrics = run_dkt_baseline(train_recs, test_recs, n_epochs=dkt_n_epochs, device=device_str)

    # ── Save ablation CSV ─────────────────────────────────────────────────────
    ablation_csv = results_base / "ablation_results.csv"
    all_keys = set()
    for r in ablation_rows:
        all_keys.update(r.keys())
    fieldnames = ["variant", "variant_display"] + sorted(
        k for k in all_keys if k not in ("variant", "variant_display")
    )

    with open(ablation_csv, "w", newline="") as f:
        f.write("# SYNTHETIC DATA — UMiKT-GAT Ablation Results\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ablation_rows)
    print(f"\nAblation CSV: {ablation_csv}")

    # ── Save baseline CSV ─────────────────────────────────────────────────────
    baseline_csv = results_base / "baseline_comparison.csv"
    full_row = next((r for r in ablation_rows if r["variant"] == "full_umikt_gat"), {})
    baseline_rows = [
        {"model": "BKT", **bkt_metrics, "data_note": "synthetic"},
        {"model": "DKT", **dkt_metrics, "data_note": "synthetic"},
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
    plot_ablation_bar_chart(ablation_rows, results_base / "ablation_bar_chart.png")
    plot_baseline_comparison(
        full_row, bkt_metrics, dkt_metrics,
        results_base / "baseline_comparison_chart.png"
    )

    # ── Write RESULTS_SUMMARY.md ──────────────────────────────────────────────
    write_results_summary(
        ablation_rows, bkt_metrics, dkt_metrics,
        results_base / "RESULTS_SUMMARY.md"
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
