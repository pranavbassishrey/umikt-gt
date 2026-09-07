"""
ASSISTments 2009-2010 Skill Builder Dataset Download Script.

This script attempts to download the ASSISTments 2009-2010 skill-builder dataset,
which is a standard public benchmark for Knowledge Tracing research.

Dataset: ASSISTments 2009-2010 Skill Builder
Source: https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data
License: Freely available for research purposes (no commercial use).
         See the ASSISTments website for full terms.
Paper: Feng, M., Heffernan, N., & Koedinger, K. (2009).

If the download fails (URL unavailable, network issue), this script falls back
to generating a structurally-equivalent synthetic dataset with the same schema,
clearly labeled as a fallback. The DATASET_CARD.md documents which layer was
actually used.

IMPORTANT: If running this script for the first time, check whether the data was
already downloaded (file existence check prevents re-download).
"""

import os
import sys
import csv
import json
import random
import math
from pathlib import Path

try:
    import urllib.request
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False


# ─── Known public download URLs for ASSISTments 2009-2010 ────────────────────
# Multiple mirrors tried in order; the script tries each and uses the first success.
ASSISTMENTS_URLS = [
    # Kaggle mirror (sometimes available without auth)
    "https://raw.githubusercontent.com/jennyzhang0215/DKVMN/master/data/assist2009.csv",
    # Alternative direct link
    "https://raw.githubusercontent.com/THUwangcy/HawkesKT/master/data/ASSISTments09/assist09.csv",
    # Backup GitHub mirror
    "https://raw.githubusercontent.com/arghosh/AKT/master/data/assist2009_pid.csv",
]

OUTPUT_DIR = Path(__file__).parent
PROCESSED_PATH = OUTPUT_DIR / "assistments09_processed.csv"
METADATA_PATH = OUTPUT_DIR / "layer1_metadata.json"


def attempt_download(url: str, dest: Path, timeout: int = 30) -> bool:
    """Attempt to download a URL to dest. Returns True on success."""
    if not HAS_URLLIB:
        return False
    try:
        print(f"  Trying: {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "UMiKT-GAT-Research/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, \
             open(dest, "wb") as out:
            out.write(resp.read())
        # Quick sanity check: file should be non-trivially sized and look like CSV
        size = dest.stat().st_size
        if size < 10000:
            dest.unlink()
            print(f"    → File too small ({size} bytes), likely an error page.")
            return False
        print(f"    → Downloaded ({size:,} bytes)")
        return True
    except Exception as e:
        print(f"    → Failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def parse_assistments_csv(raw_path: Path, processed_path: Path) -> dict:
    """
    Parse a raw ASSISTments CSV file and write a normalized processed version.

    The ASSISTments 2009-2010 format typically has columns:
      order_id, assignment_id, user_id, assistment_id, problem_id, original,
      correct, attempt_count, ms_first_response, tutor_mode, answer_type,
      sequence_id, student_class_id, hint_count, hint_total, overlap_time,
      template_id, answer_id, answer_text, first_action, bottom_hint, opportunity,
      opportunity_original, skill_id, skill_name

    We normalize to a standardized schema compatible with the UMiKT-GAT pipeline.
    """
    print(f"  Parsing {raw_path}...")

    records = []
    errors = 0
    seen_users = set()

    with open(raw_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # Handle multiple possible column name schemas across mirror variants
        user_col = next((c for c in fieldnames if "user" in c.lower()), None)
        correct_col = next((c for c in fieldnames if "correct" in c.lower()), None)
        skill_col = next((c for c in fieldnames if "skill" in c.lower() and "name" in c.lower()), None)
        if skill_col is None:
            skill_col = next((c for c in fieldnames if "skill" in c.lower()), None)
        attempt_col = next((c for c in fieldnames if "attempt" in c.lower()), None)
        time_col = next((c for c in fieldnames if "ms_first" in c.lower() or "time" in c.lower()), None)

        if not all([user_col, correct_col]):
            raise ValueError(f"Cannot find required columns in {raw_path}. "
                             f"Found: {fieldnames}")

        for row in reader:
            try:
                user_id = row.get(user_col, "").strip()
                correct_raw = row.get(correct_col, "").strip()
                skill_name = row.get(skill_col, "unknown_skill").strip() if skill_col else "unknown_skill"
                attempt_raw = row.get(attempt_col, "1").strip() if attempt_col else "1"
                time_raw = row.get(time_col, "0").strip() if time_col else "0"

                if not user_id or correct_raw not in ("0", "1"):
                    errors += 1
                    continue

                correct = int(correct_raw)
                attempts = max(1, int(attempt_raw) if attempt_raw.isdigit() else 1)
                # Convert ms to seconds, clip to reasonable range
                response_time = min(600.0, max(1.0, float(time_raw) / 1000.0)) if time_raw else 30.0

                seen_users.add(user_id)
                records.append({
                    "learner_id": user_id,
                    "skill_name": skill_name,
                    "correctness": correct,
                    "attempts": attempts,
                    "response_time_sec": round(response_time, 1),
                })
            except (ValueError, KeyError):
                errors += 1
                continue

    print(f"  Parsed {len(records):,} valid rows, {errors} skipped, "
          f"{len(seen_users):,} unique learners.")

    # Sort by learner_id to preserve sequence
    records.sort(key=lambda r: r["learner_id"])

    # Assign interaction index within each learner
    learner_idx = {}
    for rec in records:
        lid = rec["learner_id"]
        if lid not in learner_idx:
            learner_idx[lid] = 0
        rec["interaction_index"] = learner_idx[lid]
        learner_idx[lid] += 1

    # Write processed CSV
    with open(processed_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        f.write(f"# Layer 1 (real public data) — ASSISTments 2009-2010 Skill Builder\n")
        f.write(f"# Source: {raw_path.name} | {len(records):,} rows | "
                f"{len(seen_users):,} learners\n")
        f.write(f"# License: Research use; see ASSISTments website.\n")
        writer.writeheader()
        writer.writerows(records)

    return {
        "source": "ASSISTments 2009-2010 Skill Builder",
        "n_rows": len(records),
        "n_learners": len(seen_users),
        "n_skills": len(set(r["skill_name"] for r in records)),
        "n_errors_skipped": errors,
    }


def generate_fallback_layer1(processed_path: Path) -> dict:
    """
    Generate a structurally-equivalent SYNTHETIC fallback dataset that mimics
    the ASSISTments schema.

    This is used ONLY when the real dataset cannot be downloaded. The fallback
    is clearly labeled as synthetic in the file and in the metadata.

    WARNING: Using synthetic fallback means Phase 3 baseline metrics are derived
    from synthetic data, not real human performance data. This is documented in
    DATASET_CARD.md and must be disclosed in any paper using these results.
    """
    print("\n  FALLBACK: Generating synthetic Layer-1-equivalent data.")
    print("  WARNING: This is NOT the real ASSISTments dataset.")
    print("  All results from this layer will be labeled (synthetic fallback).")

    rng = random.Random(777)

    # Simulate 200 learners × 50 skills × ~20 interactions each
    # Roughly matching ASSISTments scale for a reproducibility baseline
    skill_names = [
        f"skill_{i:02d}" for i in range(50)
    ]
    n_learners = 200
    records = []
    learner_masteries = {
        lid: {s: rng.uniform(0.1, 0.6) for s in skill_names}
        for lid in range(n_learners)
    }

    for lid in range(n_learners):
        learner_id = f"fallback_learner_{lid:04d}"
        n_interactions = rng.randint(15, 40)
        for idx in range(n_interactions):
            skill = rng.choice(skill_names)
            mastery = learner_masteries[lid][skill]
            # Noisy correctness
            p = max(0.1, min(0.95, mastery + rng.gauss(0, 0.1)))
            correct = 1 if rng.random() < p else 0
            if correct:
                learner_masteries[lid][skill] = min(1.0, mastery + rng.uniform(0.05, 0.15))

            records.append({
                "learner_id": learner_id,
                "skill_name": skill,
                "correctness": correct,
                "attempts": 1,
                "response_time_sec": round(rng.uniform(5.0, 120.0), 1),
                "interaction_index": idx,
            })

    with open(processed_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        f.write("# SYNTHETIC FALLBACK — ASSISTments download failed\n")
        f.write("# This is NOT real ASSISTments data. Do not use for real benchmarking.\n")
        writer.writeheader()
        writer.writerows(records)

    return {
        "source": "SYNTHETIC_FALLBACK — ASSISTments unavailable",
        "warning": "Real ASSISTments download failed. Using synthetic fallback.",
        "n_rows": len(records),
        "n_learners": n_learners,
        "n_skills": len(skill_names),
        "is_real_data": False,
    }


def main():
    print("=" * 60)
    print("Layer 1: ASSISTments 2009-2010 Download & Processing")
    print("=" * 60)

    raw_path = OUTPUT_DIR / "assistments09_raw.csv"
    downloaded = False

    if PROCESSED_PATH.exists():
        print(f"\nProcessed file already exists: {PROCESSED_PATH}")
        print("Delete it to re-download. Exiting.")
        return

    print("\nAttempting to download ASSISTments 2009-2010 dataset...")
    for url in ASSISTMENTS_URLS:
        if attempt_download(url, raw_path):
            downloaded = True
            break

    metadata = {}

    if downloaded:
        try:
            stats = parse_assistments_csv(raw_path, PROCESSED_PATH)
            stats["download_url"] = url
            stats["is_real_data"] = True
            metadata = stats
            print(f"\nSuccess: {stats['n_rows']:,} rows from real ASSISTments data.")
        except Exception as e:
            print(f"\nParsing failed: {e}")
            downloaded = False
            if raw_path.exists():
                raw_path.unlink()

    if not downloaded:
        metadata = generate_fallback_layer1(PROCESSED_PATH)
        print(f"\nFallback written: {PROCESSED_PATH}")
        print("IMPORTANT: Metrics from Phase 3 will be labeled (synthetic fallback).")

    # Write metadata
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata: {METADATA_PATH}")
    print("\nLayer 1 data preparation complete.")


if __name__ == "__main__":
    main()
