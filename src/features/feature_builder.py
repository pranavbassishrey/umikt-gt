"""
Feature builder for the UMiKT-GAT pipeline.

Converts raw interaction dicts (from data/loader.py) into fixed-size numerical
feature vectors suitable for PyTorch training.

Feature vector layout (dimension = INPUT_DIM = 20):
  [0]   correctness (float)
  [1]   difficulty (float)
  [2]   response_time_norm (float, log-normalized)
  [3]   attempts_norm (float, normalized to [0,1])
  [4]   temporal_gap_norm (float, normalized to [0,1])
  [5]   evaluator_confidence (float)
  [6-9] question_type one-hot (4 types: mc, short, code, debug)
  [10-16] concept_id one-hot (7 concepts)

Plus optional semantic features (added when LLM extraction is used):
  [17]  misconception_severity (float, 0 if not extracted)
  [18]  misconception_confidence (float, 0 if not extracted)
  [19]  reasoning_quality (float, 0 if not extracted)

Total: INPUT_DIM = 20

The concept_id embedding is included in the feature vector (one-hot) rather
than as a separate embedding to keep the model simple and interpretable.
A learnable concept embedding could be added as a future extension.
"""

from typing import Optional

# Feature dimensionality constants (keep in sync with models)
INPUT_DIM = 20
N_CONCEPTS = 7
QUESTION_TYPES = ["multiple_choice", "short_answer", "code_completion", "debugging"]


def build_feature_vector(record: dict) -> list[float]:
    """
    Build a fixed-size feature vector from an interaction record.

    Args:
        record: Dict from loader.load_synthetic_split or compatible format.

    Returns:
        List of 20 floats in the order documented in the module docstring.
    """
    # ── Scalar features ──────────────────────────────────────────────────────
    correctness = float(record.get("correctness", 0.0))
    difficulty = float(record.get("difficulty", 0.5))
    response_time_norm = float(record.get("response_time_norm", 0.5))
    attempts_norm = float(record.get("attempts", 0.2))
    temporal_gap_norm = float(record.get("temporal_gap_norm", 0.1))
    evaluator_confidence = float(record.get("evaluator_confidence", 0.75))

    # ── Question type one-hot (dim 4) ─────────────────────────────────────────
    qt = record.get("question_type", "multiple_choice")
    qt_vec = [1.0 if qt == t else 0.0 for t in QUESTION_TYPES]

    # ── Concept ID one-hot (dim 7) ────────────────────────────────────────────
    cid = int(record.get("concept_id", 0))
    cid = max(0, min(cid, N_CONCEPTS - 1))
    cid_vec = [1.0 if cid == i else 0.0 for i in range(N_CONCEPTS)]

    # ── LLM semantic features (dim 3) — zero if not extracted ─────────────────
    mc_severity = float(record.get("misconception_severity", 0.0))
    mc_confidence = float(record.get("misconception_confidence", 0.0))
    reasoning_quality = float(record.get("reasoning_quality", 0.0))

    # ── Assemble ──────────────────────────────────────────────────────────────
    features = (
        [correctness, difficulty, response_time_norm, attempts_norm,
         temporal_gap_norm, evaluator_confidence]
        + qt_vec          # 4 dims
        + cid_vec         # 7 dims
        + [mc_severity, mc_confidence, reasoning_quality]  # 3 dims
    )

    assert len(features) == INPUT_DIM, (
        f"Feature vector length mismatch: expected {INPUT_DIM}, got {len(features)}"
    )
    return features


def build_labels(record: dict) -> dict:
    """
    Extract ground-truth labels from a record.

    For synthetic data, ground-truth labels are directly available.
    For real KT data (Layer 1), only 'correctness' is available as a label.

    Returns:
        Dict with keys:
         - 'mastery': float in [0,1]
         - 'misconception_class': int in [0,6]
         - 'misconception_severity': float in [0,1]
         - 'retention': float in [0,1]
         - 'has_full_labels': bool (True for synthetic, False for Layer 1)
    """
    if "gt_mastery" in record:
        return {
            "mastery": float(record["gt_mastery"]),
            "misconception_class": int(record["gt_misconception_class"]),
            "misconception_severity": float(record["gt_misconception_severity"]),
            "retention": float(record["gt_retention"]),
            "has_full_labels": True,
        }
    else:
        # Layer 1: only correctness is available as a proxy for mastery
        return {
            "mastery": float(record.get("correctness", 0.0)),
            "misconception_class": 0,      # unknown
            "misconception_severity": 0.0, # unknown
            "retention": float(record.get("correctness", 0.0)),  # proxy
            "has_full_labels": False,
        }
