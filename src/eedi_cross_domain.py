"""
Eedi Cross-Domain Validation Script — Task 5 (Round 4).

TIME-BOXED to ~2 hours of engineering effort. This is a supplementary
external-validity check, NOT a pipeline redesign.

Goal: Does the existing MockLLMProvider keyword/rule-based extractor, built
and tuned entirely on synthetic programming data, generalize at all to real,
human-expert-labeled misconception data in a different domain (math)?

Expectations going in:
  - Very likely to show weak or near-chance performance.
  - The extractor is a domain-specific keyword system tuned for programming.
  - Not mapping Eedi's math misconception categories to the 7-class programming
    taxonomy — only doing a binary presence/absence check.
  - An expected and legitimate finding, not a failure to hide.

Usage:
  python src/eedi_cross_domain.py

Prerequisites:
  - Kaggle CLI credentials configured (~/.kaggle/kaggle.json)
  - Run: kaggle competitions download -c eedi-mining-misconceptions-in-mathematics
    OR: place the Eedi dataset files manually in data/eedi_sample/

Stop conditions:
  - If Kaggle CLI credentials are unavailable: reports clearly and exits.
  - If the dataset format is unexpected: reports clearly and exits.
  - Does NOT attempt to retrain the model on Eedi data.
  - Does NOT build a permanent Eedi adapter into src/data/loader.py.
"""

import json
import sys
import os
import subprocess
import random
import math
from pathlib import Path
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.mock_provider import MockLLMProvider
from src.llm.schema import ExtractionRequest


EEDI_SAMPLE_DIR = Path(__file__).parent.parent / "data" / "eedi_sample"
RESULTS_PATH = Path(__file__).parent.parent / "experiments" / "results" / "eedi_cross_domain_check.json"
SAMPLE_SIZE = 200
SAMPLE_SEED = 42


def check_kaggle_credentials() -> bool:
    """Check whether Kaggle credentials are available."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True
    # Also check env vars
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return False


def try_download_eedi() -> bool:
    """
    Attempt to download the Eedi dataset via Kaggle CLI.
    Returns True if successful, False otherwise.
    """
    EEDI_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    print("  Attempting kaggle CLI download...")
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "download",
             "-c", "eedi-mining-misconceptions-in-mathematics",
             "-p", str(EEDI_SAMPLE_DIR)],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            print("  Kaggle download succeeded.")
            return True
        else:
            print(f"  Kaggle CLI error: {result.stderr.strip()}")
            return False
    except FileNotFoundError:
        print("  Kaggle CLI not installed (kaggle command not found).")
        return False
    except subprocess.TimeoutExpired:
        print("  Kaggle download timed out.")
        return False
    except Exception as e:
        print(f"  Kaggle download error: {e}")
        return False


def find_eedi_csv() -> Path | None:
    """Find the Eedi dataset CSV file (may be zipped or already extracted)."""
    import zipfile

    # Try to unzip any downloaded zip files
    for zf in EEDI_SAMPLE_DIR.glob("*.zip"):
        print(f"  Unzipping: {zf.name}")
        try:
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(EEDI_SAMPLE_DIR)
        except Exception as e:
            print(f"  Unzip error: {e}")

    # Look for CSV files — prefer files with "question" or "train" in name
    candidates = list(EEDI_SAMPLE_DIR.glob("**/*.csv"))
    if not candidates:
        return None

    # Prefer files that seem like the main question file
    for candidate in candidates:
        name = candidate.name.lower()
        if "question" in name or "train" in name:
            return candidate

    return candidates[0]  # fallback: first CSV found


def load_eedi_questions(csv_path: Path) -> list[dict]:
    """
    Load Eedi question records. Expected columns include:
      QuestionId, SubjectId, ConstructId, CorrectAnswer,
      AnswerAText, AnswerBText, AnswerCText, AnswerDText,
      QuestionText, MisconceptionAId, MisconceptionBId,
      MisconceptionCId, MisconceptionDId (or similar)
    Returns list of dicts.
    """
    import csv
    rows = []
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def extract_eedi_ground_truth(row: dict) -> bool | None:
    """
    Extract ground-truth "has expert-labeled misconception" signal.

    Eedi's annotation: Each wrong answer (A/B/C/D, excluding the correct answer)
    may have a MisconceptionXId. If ANY non-correct answer has a non-empty,
    non-null MisconceptionId, we label this question as "has_misconception=True".

    Returns True/False, or None if we can't determine.
    """
    correct = row.get("CorrectAnswer", "").strip().upper()
    has_any_misconception = False

    for letter in ["A", "B", "C", "D"]:
        if letter == correct:
            continue  # skip the correct answer
        mc_col = f"Misconception{letter}Id"
        mc_val = row.get(mc_col, "").strip()
        if mc_val and mc_val.lower() not in ("", "nan", "none", "null"):
            has_any_misconception = True
            break

    return has_any_misconception


def build_eedi_student_answer_text(row: dict) -> str:
    """
    Build a "student answer text" from Eedi's distractor answers.

    Since Eedi questions have 4 answer choices, we concatenate all incorrect
    answers into a text the extractor can analyze. We use the text of wrong
    answers because those represent the misconception-containing responses.
    """
    correct = row.get("CorrectAnswer", "").strip().upper()
    parts = []
    for letter in ["A", "B", "C", "D"]:
        if letter == correct:
            continue
        col = f"Answer{letter}Text"
        text = row.get(col, "").strip()
        if text:
            parts.append(text)
    return " | ".join(parts) if parts else row.get("QuestionText", "")


def run_eedi_check():
    """
    Main function: download, sample, extract, and report.
    """
    print("=" * 60)
    print("EEDI CROSS-DOMAIN VALIDATION CHECK — Task 5 (Round 4)")
    print("Supplementary check. Time-boxed. Expected: near-chance performance.")
    print("=" * 60)

    EEDI_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Obtain Eedi data ─────────────────────────────────────────────
    csv_path = find_eedi_csv()
    if csv_path is None:
        if not check_kaggle_credentials():
            msg = (
                "Kaggle CLI credentials not found (~/.kaggle/kaggle.json or env vars).\n"
                "Task 5 cannot proceed. Reporting this stop condition as required."
            )
            print(f"\n  STOP: {msg}")
            result = {
                "status": "stopped_no_kaggle_credentials",
                "sample_size": 0,
                "extraction_method": "MockLLMProvider keyword/regex extractor",
                "agreement_statistic": None,
                "interpretation": msg,
                "note": "Task 5 stop condition: Kaggle credentials unavailable. No fabrication.",
            }
            with open(RESULTS_PATH, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Result saved: {RESULTS_PATH}")
            return result

        # Try download
        downloaded = try_download_eedi()
        if not downloaded:
            msg = "Kaggle download failed. See terminal output for details."
            print(f"\n  STOP: {msg}")
            result = {
                "status": "stopped_download_failed",
                "sample_size": 0,
                "extraction_method": "MockLLMProvider keyword/regex extractor",
                "agreement_statistic": None,
                "interpretation": msg,
                "note": "Task 5 stop condition: Kaggle download failed. No fabrication.",
            }
            with open(RESULTS_PATH, "w") as f:
                json.dump(result, f, indent=2)
            return result

        csv_path = find_eedi_csv()
        if csv_path is None:
            msg = "Could not locate a CSV file in the downloaded Eedi data."
            print(f"\n  STOP: {msg}")
            result = {
                "status": "stopped_csv_not_found",
                "sample_size": 0,
                "extraction_method": "MockLLMProvider keyword/regex extractor",
                "agreement_statistic": None,
                "interpretation": msg,
                "note": "Task 5 stop condition: CSV not found after download. No fabrication.",
            }
            with open(RESULTS_PATH, "w") as f:
                json.dump(result, f, indent=2)
            return result

    print(f"\n  Using Eedi CSV: {csv_path}")

    # ── Step 2: Load and sample 200 questions ────────────────────────────────
    all_rows = load_eedi_questions(csv_path)
    print(f"  Total questions in dataset: {len(all_rows)}")

    if len(all_rows) < SAMPLE_SIZE:
        print(f"  WARNING: Only {len(all_rows)} rows available; using all.")
        sample = all_rows
    else:
        rng = random.Random(SAMPLE_SEED)
        sample = rng.sample(all_rows, SAMPLE_SIZE)
    print(f"  Sampled: {len(sample)} questions (seed={SAMPLE_SEED})")

    # Save subsample with clear provenance header
    import csv as _csv
    sample_csv_path = EEDI_SAMPLE_DIR / "eedi_200_sample.csv"
    if sample:
        fieldnames = list(sample[0].keys())
        with open(sample_csv_path, "w", newline="", encoding="utf-8") as f:
            f.write("# Eedi cross-domain validation sample\n")
            f.write("# Source: Eedi - Mining Misconceptions in Mathematics (Kaggle)\n")
            f.write("# License: Apache 2.0 (Kaggle competition dataset)\n")
            f.write(f"# Sample: {len(sample)} questions, seed={SAMPLE_SEED}\n")
            f.write("# Used for: UMiKT-GAT Round 4 cross-domain validation (supplementary check only)\n")
            f.write("# NOTE: This is a subsample only. The full dataset belongs to Eedi/NeurIPS 2024.\n")
            writer = _csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(sample)
        print(f"  Subsample saved: {sample_csv_path}")

    # ── Step 3: Extract misconceptions using MockLLMProvider ─────────────────
    provider = MockLLMProvider()
    extractor_flags = []     # 1 if extractor flagged non-none misconception, 0 otherwise
    gt_flags = []            # 1 if Eedi has expert-labeled misconception, 0 otherwise
    skipped = 0

    for row in sample:
        # Ground truth: does Eedi have a specific expert-labeled misconception?
        gt = extract_eedi_ground_truth(row)
        if gt is None:
            skipped += 1
            continue

        # Student answer text: wrong answer option texts
        student_text = build_eedi_student_answer_text(row)
        if not student_text.strip():
            skipped += 1
            continue

        # Run the same MockLLMProvider extractor (unchanged from rest of pipeline)
        req = ExtractionRequest(
            student_answer=student_text,
            question=row.get("QuestionText", ""),
            concept_name="math",  # placeholder — no taxonomy mapping attempted
            concept_id=0,         # placeholder
            expected_answer=None,
        )
        try:
            result_dict = provider._extract_raw(req)
            mc_label = result_dict.get("misconception_label", "none")
            extractor_flagged = 1 if mc_label != "none" else 0
        except Exception:
            extractor_flagged = 0

        extractor_flags.append(extractor_flagged)
        gt_flags.append(1 if gt else 0)

    n_valid = len(extractor_flags)
    print(f"\n  Valid samples processed: {n_valid} (skipped: {skipped})")

    if n_valid == 0:
        msg = "No valid samples could be processed (all skipped)."
        print(f"  STOP: {msg}")
        result = {
            "status": "no_valid_samples",
            "sample_size": SAMPLE_SIZE,
            "n_valid": 0,
            "extraction_method": "MockLLMProvider keyword/regex extractor",
            "agreement_statistic": None,
            "interpretation": msg,
        }
        with open(RESULTS_PATH, "w") as f:
            json.dump(result, f, indent=2)
        return result

    # ── Step 4: Compute agreement statistic ──────────────────────────────────
    import numpy as np

    ext_arr = np.array(extractor_flags)
    gt_arr = np.array(gt_flags)

    # Point-biserial correlation: extractor flag (binary) vs. GT flag (binary)
    # (degenerate case of Pearson r for binary-binary data = phi coefficient)
    try:
        if gt_arr.std() == 0 or ext_arr.std() == 0:
            # All same class — can't compute correlation
            pbis_r = float("nan")
            pbis_p = float("nan")
        else:
            pbis_r, pbis_p = scipy_stats.pointbiserialr(gt_arr, ext_arr)
            pbis_r = float(pbis_r)
            pbis_p = float(pbis_p)
    except Exception as e:
        pbis_r = float("nan")
        pbis_p = float("nan")

    # Simple agreement / accuracy: extractor says misconception present ↔ Eedi GT says so
    correct = int(np.sum(ext_arr == gt_arr))
    accuracy = correct / n_valid

    # Baseline: always-positive predictor accuracy (since most Eedi wrong answers have a MC)
    baseline_always_pos = float(gt_arr.mean())
    baseline_always_neg = 1.0 - baseline_always_pos
    chance_baseline = max(baseline_always_pos, baseline_always_neg)

    gt_positive_rate = float(gt_arr.mean())
    extractor_positive_rate = float(ext_arr.mean())

    # ── Step 5: Honest interpretation ────────────────────────────────────────
    if math.isnan(pbis_r):
        corr_str = "N/A (degenerate class distribution)"
    else:
        corr_str = f"point-biserial r = {pbis_r:.4f}, p = {pbis_p:.4f}"

    interpretation = (
        f"the extraction approach performs at approximately {accuracy:.3f} presence/absence accuracy "
        f"and {corr_str}, compared to {chance_baseline:.3f} chance/majority-class baseline level, "
        f"when applied out-of-domain to real human-annotated math misconception data. "
        f"The extractor flags {extractor_positive_rate:.1%} of samples as misconceptions while "
        f"Eedi's ground truth has {gt_positive_rate:.1%} as misconception-present. "
        f"This is expected: the extractor is a domain-specific keyword system tuned for "
        f"programming-domain vocabulary, not mathematical misconception patterns."
    )

    print(f"\n  Ground-truth positive rate (Eedi): {gt_positive_rate:.3f}")
    print(f"  Extractor positive rate:           {extractor_positive_rate:.3f}")
    print(f"  Presence/absence accuracy:         {accuracy:.4f}")
    print(f"  Chance (majority-class) baseline:  {chance_baseline:.4f}")
    print(f"  Point-biserial correlation:        {pbis_r:.4f} (p={pbis_p:.4f})" if not math.isnan(pbis_r) else "  Point-biserial: N/A (degenerate)")
    print(f"\n  Interpretation: {interpretation}")

    result = {
        "status": "completed",
        "sample_size": SAMPLE_SIZE,
        "n_valid": n_valid,
        "n_skipped": skipped,
        "sample_seed": SAMPLE_SEED,
        "sample_csv": str(sample_csv_path),
        "eedi_source_csv": str(csv_path),
        "extraction_method": "MockLLMProvider keyword/regex extractor (unchanged from pipeline)",
        "note_no_taxonomy_mapping": (
            "The 7-class programming taxonomy was NOT mapped to Eedi's math misconception "
            "categories. Only binary presence/absence was compared."
        ),
        "gt_positive_rate": gt_positive_rate,
        "extractor_positive_rate": extractor_positive_rate,
        "presence_absence_accuracy": accuracy,
        "chance_majority_baseline": chance_baseline,
        "point_biserial_r": pbis_r,
        "point_biserial_p": pbis_p,
        "agreement_statistic": corr_str,
        "interpretation": interpretation,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved: {RESULTS_PATH}")
    return result


if __name__ == "__main__":
    run_eedi_check()
