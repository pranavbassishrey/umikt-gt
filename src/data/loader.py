"""
Data loader and preprocessing for the UMiKT-GAT pipeline.

Handles both:
 - Layer 1: Public KT dataset (ASSISTments 2009-2010 or fallback)
 - Layer 2: Synthetic Programming Fundamentals dataset

Design principle: All numerical features are normalized here. All concept_id
mappings are validated against concept_graph.json. The output is a consistent
dictionary format consumed by feature builders.
"""

import csv
import json
import os
from pathlib import Path
from typing import Optional

# Concept ID range for synthetic dataset
N_CONCEPTS = 7

# ─── Numeric feature columns in synthetic dataset ───────────────────────────
SYNTHETIC_FEATURE_COLS = [
    "correctness",
    "difficulty",
    "response_time_sec",
    "attempts",
    "concept_id",
    "temporal_gap_hours",
    "evaluator_confidence",
]

SYNTHETIC_LABEL_COLS = [
    "gt_mastery",
    "gt_misconception_class",
    "gt_misconception_severity",
    "gt_retention",
]


def _normalize_response_time(t_sec: float) -> float:
    """Log-normalize response time to [0,1] range.

    Rationale: response time is log-normally distributed (see generate_synthetic.py).
    We use log(1+t)/log(1+600) so the range [1,600] maps to approximately [0,1].
    """
    import math
    return math.log(1 + t_sec) / math.log(1 + 600.0)


def _read_csv_skip_comments(path: Path):
    """Read a CSV file, skipping lines that start with '#'."""
    lines = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("#"):
                lines.append(line)
    import io
    reader = csv.DictReader(io.StringIO("".join(lines)))
    return list(reader)


def load_synthetic_split(split: str, data_dir: Optional[Path] = None) -> list[dict]:
    """
    Load a synthetic dataset split (train / val / test).

    Args:
        split: One of 'train', 'val', 'test'.
        data_dir: Path to layer2_synthetic directory. Defaults to relative path.

    Returns:
        List of interaction dicts with normalized numerical features
        and ground-truth labels.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data" / "layer2_synthetic"

    path = data_dir / f"synthetic_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Synthetic {split} split not found at {path}. "
            "Run data/layer2_synthetic/generate_synthetic.py first."
        )

    rows = _read_csv_skip_comments(path)
    records = []

    for row in rows:
        try:
            concept_id = int(row["concept_id"])
            if concept_id < 0 or concept_id >= N_CONCEPTS:
                continue

            rec = {
                "learner_id": row["learner_id"],
                "interaction_id": int(row["interaction_id"]),
                "concept_id": concept_id,
                "concept_name": row["concept_name"],
                # Normalized numerical features
                "correctness": float(row["correctness"]),
                "difficulty": float(row["difficulty"]),
                "response_time_norm": _normalize_response_time(float(row["response_time_sec"])),
                "attempts": min(float(row["attempts"]) / 5.0, 1.0),  # normalize to [0,1]
                "temporal_gap_norm": min(float(row["temporal_gap_hours"]) / 24.0, 1.0),
                "evaluator_confidence": float(row["evaluator_confidence"]),
                # Question type one-hot (4 types)
                "question_type": row["question_type"],
                # Ground-truth labels (for training/evaluation)
                "gt_mastery": float(row["gt_mastery"]),
                "gt_misconception_class": int(row["gt_misconception_class"]),
                "gt_misconception_severity": float(row["gt_misconception_severity"]),
                "gt_retention": float(row["gt_retention"]),
                # Raw values for analysis
                "response_time_sec": float(row["response_time_sec"]),
                "temporal_gap_hours": float(row["temporal_gap_hours"]),
                "sim_hour": float(row["sim_hour"]),
            }
            records.append(rec)
        except (KeyError, ValueError):
            continue

    return records


def group_by_learner(records: list[dict]) -> dict[str, list[dict]]:
    """
    Group interaction records by learner_id, preserving temporal order.

    Args:
        records: Flat list of interaction dicts.

    Returns:
        Dict mapping learner_id → sorted list of interactions.
    """
    groups: dict[str, list[dict]] = {}
    for rec in records:
        lid = rec["learner_id"]
        if lid not in groups:
            groups[lid] = []
        groups[lid].append(rec)

    # Sort each learner's interactions by interaction_id
    for lid in groups:
        groups[lid].sort(key=lambda r: r["interaction_id"])

    return groups


def load_layer1_split(data_dir: Optional[Path] = None) -> list[dict]:
    """
    Load the Layer 1 (ASSISTments or fallback) dataset.

    Returns a list of records in a standardized schema.
    The 'is_synthetic_fallback' field is True if the real dataset was unavailable.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data" / "layer1_public_kt"

    processed_path = data_dir / "assistments09_processed.csv"
    metadata_path = data_dir / "layer1_metadata.json"

    if not processed_path.exists():
        raise FileNotFoundError(
            f"Layer 1 data not found at {processed_path}. "
            "Run data/layer1_public_kt/download_assistments.py first."
        )

    is_fallback = False
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        is_fallback = not meta.get("is_real_data", True)

    rows = _read_csv_skip_comments(processed_path)
    records = []

    for row in rows:
        try:
            rec = {
                "learner_id": row["learner_id"],
                "skill_name": row["skill_name"],
                "correctness": float(row["correctness"]),
                "attempts": min(float(row["attempts"]) / 5.0, 1.0),
                "response_time_norm": _normalize_response_time(float(row["response_time_sec"])),
                "interaction_index": int(row["interaction_index"]),
                "is_synthetic_fallback": is_fallback,
            }
            records.append(rec)
        except (KeyError, ValueError):
            continue

    return records


def make_layer1_train_val_test(
    records: list[dict],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split Layer 1 records by learner into train/val/test.

    Splitting by learner (not by row) prevents data leakage — each learner
    appears in only one split, preserving sequential dependencies within splits.

    Args:
        records: Flat list of interaction records.
        val_frac: Fraction of learners for validation.
        test_frac: Fraction of learners for test.
        seed: Random seed.

    Returns:
        (train_records, val_records, test_records)
    """
    import random
    rng = random.Random(seed)

    learner_ids = sorted(set(r["learner_id"] for r in records))
    rng.shuffle(learner_ids)

    n = len(learner_ids)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)

    test_ids = set(learner_ids[:n_test])
    val_ids = set(learner_ids[n_test:n_test + n_val])
    train_ids = set(learner_ids[n_test + n_val:])

    train = [r for r in records if r["learner_id"] in train_ids]
    val = [r for r in records if r["learner_id"] in val_ids]
    test = [r for r in records if r["learner_id"] in test_ids]

    return train, val, test


def load_concept_graph(graph_path: Optional[Path] = None) -> dict:
    """Load the concept graph JSON."""
    if graph_path is None:
        graph_path = Path(__file__).parent.parent.parent / "data" / "concept_graph.json"
    with open(graph_path) as f:
        return json.load(f)


def load_handwritten_qa(data_dir: Optional[Path] = None) -> list[dict]:
    """Load the hand-written Q&A pairs for LLM extraction evaluation."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data" / "layer2_handwritten"
    path = data_dir / "qa_pairs.json"
    with open(path) as f:
        return json.load(f)
