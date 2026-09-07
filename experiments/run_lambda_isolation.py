"""
Run Issue B Lambda Isolation Study:
Compare Config A, Config B, and Config C for full UMiKT-GAT (50 epochs each).

Config A (Round 1 original) : lambda_mastery=1.0, lambda_mc=0.5,  lambda_ret=0.3
Config B (lambda_mc only)   : lambda_mastery=1.0, lambda_mc=0.10, lambda_ret=0.3
Config C (Round 2 both)     : lambda_mastery=1.0, lambda_mc=0.10, lambda_ret=1.0

Metrics:
- Mastery AUC
- Mastery RMSE
- Mastery Accuracy (using validation-derived threshold)
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_synthetic_split, group_by_learner
from src.models.umikt_gat import make_ablation_config
from src.train import train, set_seed, evaluate
from src.models.umikt_gat import UMiKTGATModel

CONFIGS = [
    {
        "name": "Config A (Round 1 original)",
        "id": "config_a",
        "lambda_mastery": 1.0,
        "lambda_misconception": 0.5,
        "lambda_retention": 0.3,
    },
    {
        "name": "Config B (lambda_mc only)",
        "id": "config_b",
        "lambda_mastery": 1.0,
        "lambda_misconception": 0.10,
        "lambda_retention": 0.3,
    },
    {
        "name": "Config C (Round 2 both)",
        "id": "config_c",
        "lambda_mastery": 1.0,
        "lambda_misconception": 0.10,
        "lambda_retention": 1.0,
    },
]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_base = Path(__file__).parent.parent / "experiments" / "results" / "lambda_isolation"
    results_base.mkdir(parents=True, exist_ok=True)

    val_recs = load_synthetic_split("val")
    test_recs = load_synthetic_split("test")
    val_by_learner = group_by_learner(val_recs)
    test_by_learner = group_by_learner(test_recs)

    all_results = []

    print("=" * 60)
    print("RUNNING ISSUE B: LAMBDA WEIGHT ISOLATION STUDY (50 EPOCHS)")
    print("=" * 60)

    for item in CONFIGS:
        cid = item["id"]
        cname = item["name"]
        print(f"\n{'─'*50}")
        print(f"Running {cname}...")
        print(f"  λ_mastery={item['lambda_mastery']}, λ_mc={item['lambda_misconception']}, λ_ret={item['lambda_retention']}")

        set_seed(42)
        cfg = make_ablation_config("full_umikt_gat")
        cfg["lambda_mastery"] = item["lambda_mastery"]
        cfg["lambda_misconception"] = item["lambda_misconception"]
        cfg["lambda_retention"] = item["lambda_retention"]
        cfg["n_epochs"] = 50
        cfg["patience"] = 10

        t0 = time.perf_counter()
        variant_subpath = f"lambda_isolation/{cid}"
        train(cfg, variant_name=variant_subpath)
        elapsed = time.perf_counter() - t0

        ckpt_path = Path(__file__).parent.parent / "experiments" / "results" / variant_subpath / "best_model.pt"
        model = UMiKTGATModel(cfg).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        # 1. Validation threshold
        val_metrics = evaluate(model, val_by_learner, cfg, device)
        val_pred_thresh = val_metrics["pred_thresh"]

        # 2. Test evaluation using val-derived threshold
        test_metrics = evaluate(model, test_by_learner, cfg, device, pred_thresh=val_pred_thresh)

        res = {
            "config": item["name"],
            "id": cid,
            "lambda_mastery": item["lambda_mastery"],
            "lambda_mc": item["lambda_misconception"],
            "lambda_ret": item["lambda_retention"],
            "val_pred_thresh": val_pred_thresh,
            "pred_bin_counts": test_metrics["pred_bin_counts"],
            "mastery_auc": test_metrics["mastery_auc"],
            "mastery_rmse": test_metrics["mastery_rmse"],
            "mastery_accuracy": test_metrics["mastery_accuracy"],
            "elapsed_sec": round(elapsed, 2),
        }
        all_results.append(res)
        print(f"  Finished {cid}: AUC={res['mastery_auc']:.4f}, RMSE={res['mastery_rmse']:.4f}, Acc={res['mastery_accuracy']:.4f}, Thresh={val_pred_thresh:.6f}, counts={res['pred_bin_counts']}")

    summary_file = results_base / "isolation_results.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("LAMBDA ISOLATION STUDY SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Config':<30} | {'λ_mc':<6} | {'λ_ret':<6} | {'AUC':<8} | {'RMSE':<8} | {'Accuracy':<8} | {'Thresh':<8} | {'pred_bin'}")
    print("-" * 95)
    for r in all_results:
        print(f"{r['config']:<30} | {r['lambda_mc']:<6.2f} | {r['lambda_ret']:<6.2f} | {r['mastery_auc']:<8.4f} | {r['mastery_rmse']:<8.4f} | {r['mastery_accuracy']:<8.4f} | {r['val_pred_thresh']:<8.4f} | {r['pred_bin_counts']}")
    print("=" * 95)

if __name__ == "__main__":
    main()
