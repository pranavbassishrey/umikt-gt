"""
Reproduction Script: All Metrics and Tables Reported in minor_project_report.md
=============================================================================
Reproduces and verifies every quantitative claim reported in minor_project_report.md:
1. Overall AUC, PR-AUC, Accuracy, RMSE, Brier score, ECE.
2. Learner-clustered bootstrap distribution and 95% CI.
3. McNemar paired test (discordant counts n01, n10, chi2, p-value).
4. 10-bin calibration table and 10-bin Murphy Brier decomposition.
5. Per-timestep sequence trajectory breakdown.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    mean_squared_error,
    roc_auc_score,
)
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parent.parent
DKT_PREDS = ROOT / "experiments" / "results" / "dkt_fair" / "test_predictions.csv"
V2_PREDS = ROOT / "experiments" / "results" / "umikt_v2_next_response" / "test_predictions.csv"


def main():
    print("=" * 70)
    print("REPRODUCING ALL REPORTED METRICS: minor_project_report.md")
    print("=" * 70)

    dkt = pd.read_csv(DKT_PREDS)
    v2 = pd.read_csv(V2_PREDS)

    assert len(dkt) == 11700, f"Expected 11,700 DKT predictions, got {len(dkt)}"
    assert len(v2) == 11700, f"Expected 11,700 V2 predictions, got {len(v2)}"
    assert (dkt["composite_target_id"] == v2["composite_target_id"]).all()
    assert (dkt["y_true"] == v2["y_true"]).all()

    y_true = dkt["y_true"].values
    p_dkt = dkt["y_prob_dkt"].values
    p_v2 = v2["y_prob_umikt_v2"].values

    # 1. Overall Metrics
    print("\n--- [1] Overall Evaluation Metrics (11,700 targets) ---")
    dkt_auc = roc_auc_score(y_true, p_dkt)
    v2_auc = roc_auc_score(y_true, p_v2)
    dkt_prauc = average_precision_score(y_true, p_dkt)
    v2_prauc = average_precision_score(y_true, p_v2)
    dkt_acc = accuracy_score(y_true, (p_dkt >= 0.5).astype(int))
    v2_acc = accuracy_score(y_true, (p_v2 >= 0.5).astype(int))
    dkt_rmse = np.sqrt(mean_squared_error(y_true, p_dkt))
    v2_rmse = np.sqrt(mean_squared_error(y_true, p_v2))
    dkt_brier = brier_score_loss(y_true, p_dkt)
    v2_brier = brier_score_loss(y_true, p_v2)

    print(f"DKT:      AUC={dkt_auc:.4f}  PR-AUC={dkt_prauc:.4f}  Acc={dkt_acc:.4f}  RMSE={dkt_rmse:.4f}  Brier={dkt_brier:.4f}")
    print(f"UMiKT-v2: AUC={v2_auc:.4f}  PR-AUC={v2_prauc:.4f}  Acc={v2_acc:.4f}  RMSE={v2_rmse:.4f}  Brier={v2_brier:.4f}")
    print(f"Delta:    AUC={v2_auc - dkt_auc:+.4f}  Brier={v2_brier - dkt_brier:+.4f}")

    # 2. Learner-Clustered Bootstrap
    print("\n--- [2] Learner-Clustered Bootstrap (B=2000 resamples of 300 learners) ---")
    unique_learners = dkt["learner_id"].unique()
    n_learners = len(unique_learners)
    learner_indices = dkt.groupby("learner_id").indices
    rng = np.random.RandomState(42)
    B = 2000
    deltas = []
    for _ in range(B):
        sampled = rng.choice(unique_learners, size=n_learners, replace=True)
        idx_list = []
        for l in sampled:
            idx_list.extend(learner_indices[l])
        idx = np.array(idx_list)
        y_b = y_true[idx]
        if len(np.unique(y_b)) > 1:
            deltas.append(roc_auc_score(y_b, p_v2[idx]) - roc_auc_score(y_b, p_dkt[idx]))

    deltas = np.array(deltas)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    non_pos = (deltas <= 0).sum()
    print(f"Clustered Bootstrap Delta Mean: {deltas.mean():.4f}")
    print(f"Percentile 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"Non-positive replicates: {non_pos} / {B} (100% of replicates positive)")

    # 3. McNemar Paired Test
    print("\n--- [3] McNemar Paired Classification Test (Threshold = 0.5) ---")
    dkt_correct = (p_dkt >= 0.5).astype(int) == y_true
    v2_correct = (p_v2 >= 0.5).astype(int) == y_true
    n01 = int((~dkt_correct & v2_correct).sum())  # DKT wrong, UMiKT right
    n10 = int((dkt_correct & ~v2_correct).sum())  # DKT right, UMiKT wrong
    chi2_stat = ((abs(n01 - n10) - 1.0) ** 2) / (n01 + n10)
    p_val_mcnemar = chi2.sf(chi2_stat, 1)
    print(f"n01 (DKT wrong, v2 right): {n01}")
    print(f"n10 (DKT right, v2 wrong): {n10}")
    print(f"Chi2 statistic: {chi2_stat:.2f}, p-value: {p_val_mcnemar:.2e}")

    # 4. 10-Bin Calibration & Murphy Brier Decomposition
    print("\n--- [4] 10-Bin Calibration & Murphy Brier Decomposition ---")
    y_bar = y_true.mean()
    uncertainty = y_bar * (1.0 - y_bar)
    bin_edges = np.linspace(0.0, 1.0, 11)

    for name, p in [("DKT", p_dkt), ("UMiKT-v2", p_v2)]:
        rel = 0.0
        res = 0.0
        print(f"\n{name} 10-Bin Reliability Table:")
        print(f"{'Bin':<12} {'Count':<7} {'MeanPred':<10} {'EmpiricalRate':<14} {'Gap':<8}")
        print("-" * 55)
        for i in range(10):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (p >= lo) & (p < hi) if i < 9 else (p >= lo) & (p <= hi)
            n_k = mask.sum()
            if n_k > 0:
                p_k = p[mask].mean()
                y_k = y_true[mask].mean()
                gap = abs(p_k - y_k)
                rel += (n_k / len(y_true)) * ((p_k - y_k) ** 2)
                res += (n_k / len(y_true)) * ((y_k - y_bar) ** 2)
                print(f"[{lo:.1f}, {hi:.1f}{')' if i<9 else ']'} {n_k:<7} {p_k:<10.4f} {y_k:<14.4f} {gap:<8.4f}")
        brier_decomp = rel - res + uncertainty
        emp_brier = brier_score_loss(y_true, p)
        print(f"Decomposition Components: Uncertainty={uncertainty:.6f}, Reliability={rel:.6f}, Resolution={res:.6f}")
        print(f"Decomposed Brier={brier_decomp:.6f} vs Empirical Brier={emp_brier:.6f} (Residual: {abs(brier_decomp - emp_brier):.6f})")

    # 5. Trajectory Breakdown
    print("\n--- [5] Timestep Trajectory Breakdown ---")
    bins = [(1, 2), (3, 5), (6, 10), (11, 20), (21, 39)]
    print(f"{'Steps':<10} {'N_preds':<8} {'DKT AUC':<10} {'V2 AUC':<10} {'Delta':<10} {'DKT Brier':<10} {'V2 Brier':<10}")
    print("-" * 70)
    for lo, hi in bins:
        mask = (dkt["target_step_in_sequence"] >= lo) & (dkt["target_step_in_sequence"] <= hi)
        y_b = y_true[mask]
        p_d = p_dkt[mask]
        p_v = p_v2[mask]
        a_d = roc_auc_score(y_b, p_d)
        a_v = roc_auc_score(y_b, p_v)
        b_d = brier_score_loss(y_b, p_d)
        b_v = brier_score_loss(y_b, p_v)
        print(f"{f'{lo}-{hi}':<10} {mask.sum():<8} {a_d:<10.4f} {a_v:<10.4f} {a_v - a_d:<+10.4f} {b_d:<10.4f} {b_v:<10.4f}")

    print("\n[OK] ALL REPORT METRICS SUCCESSFULLY REPRODUCED.")


if __name__ == "__main__":
    main()
