"""
Process ASSISTments / FoundationalASSIST Dataset for UMiKT-GAT and Knowledge Tracing.

Loads the dataset using Hugging Face's `datasets` library:
  - 'Foundational ASSIST Dataset' (Problems)
  - 'Skills'
  - 'Interactions'

Extracts and normalizes student interaction trajectories, maps Common Core
standards into the 7-node pedagogical concept graph, and creates train/val/test
splits grouped by learner to avoid data leakage.
"""

import os
import sys
import math
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Domain to concept_id mapping (7 concepts)
DOMAIN_TO_CONCEPT = {
    "ARITH": 0, "OA": 0, "NBT": 0, "MD": 0, "NF": 0,
    "NS": 1,
    "RP": 2,
    "EE": 3,
    "F": 4,
    "G": 5,
    "SP": 6,
}

CONCEPT_NAMES = {
    0: "Arithmetic & Operations",
    1: "The Number System",
    2: "Ratios & Proportions",
    3: "Expressions & Equations",
    4: "Functions",
    5: "Geometry",
    6: "Statistics & Probability",
}


def map_node_code_to_concept(node_code: str) -> tuple[int, str]:
    """Map a Common Core node code (e.g. '7.RP.A.3', '6.NS.A.1') to concept_id."""
    if not isinstance(node_code, str) or not node_code.strip():
        return 0, CONCEPT_NAMES[0]

    parts = node_code.strip().split(".")
    domain = parts[1] if len(parts) >= 2 else parts[0]
    domain_upper = domain.upper()

    cid = DOMAIN_TO_CONCEPT.get(domain_upper, 0)
    return cid, CONCEPT_NAMES[cid]


def map_problem_type(prob_type: str) -> str:
    """Normalize problem type to one of the 4 standard types in feature builder."""
    if not isinstance(prob_type, str):
        return "short_answer"
    pt = prob_type.lower()
    if "multiple" in pt or "choice" in pt:
        return "multiple_choice"
    elif "fill" in pt or "numeric" in pt:
        return "short_answer"
    elif "code" in pt:
        return "code_completion"
    else:
        return "debugging"


def load_dataset_bundle(data_source: str = "auto"):
    """
    Load FoundationalASSIST via huggingface `datasets`.
    Supports loading from HF Hub or local clone directory.
    """
    from datasets import load_dataset

    repo_or_dir = None
    if data_source == "auto":
        # Check local folder first
        local_dir = PROJECT_ROOT / "FoundationalASSIST"
        if (local_dir / "Data" / "Interactions.csv").exists():
            repo_or_dir = str(local_dir)
        else:
            repo_or_dir = "ASSISTments/FoundationalASSIST"
    else:
        repo_or_dir = data_source

    print(f"Loading FoundationalASSIST dataset from: {repo_or_dir}")
    try:
        ds_problems = load_dataset(repo_or_dir, "Foundational ASSIST Dataset")["train"]
        ds_skills = load_dataset(repo_or_dir, "Skills")["train"]
        ds_interactions = load_dataset(repo_or_dir, "Interactions")["train"]
    except Exception as e:
        print(f"Direct HF load raised: {e}. Falling back to direct CSV reading...")
        data_dir = PROJECT_ROOT / "FoundationalASSIST" / "Data"
        df_problems = pd.read_csv(data_dir / "Problems.csv")
        df_skills = pd.read_csv(data_dir / "Skills.csv")
        df_interactions = pd.read_csv(data_dir / "Interactions.csv")
        return df_problems, df_skills, df_interactions

    df_problems = ds_problems.to_pandas()
    df_skills = ds_skills.to_pandas()
    df_interactions = ds_interactions.to_pandas()
    return df_problems, df_skills, df_interactions


def process_dataset(
    max_learners: int = 2500,
    max_seq_len: int = 50,
    min_interactions_per_learner: int = 15,
    output_dir: Path | None = None,
):
    """
    Process FoundationalASSIST into normalized interaction trajectories.
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "foundational_assist"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_problems, df_skills, df_interactions = load_dataset_bundle()

    print(f"Loaded: {len(df_problems):,} problems, {len(df_skills):,} skill mappings, "
          f"{len(df_interactions):,} interactions.")

    # 1. Clean interactions
    df_interactions = df_interactions.dropna(subset=["user_id", "problem_id", "discrete_score"]).copy()
    df_interactions = df_interactions[df_interactions["discrete_score"].isin([0.0, 1.0, 0, 1])].copy()
    df_interactions["correctness"] = df_interactions["discrete_score"].astype(float)
    df_interactions["hint_count"] = df_interactions["hint_count"].fillna(0).astype(int)
    # Attempts: 1 base try + hint_count (capped at 5)
    df_interactions["attempts"] = (1 + df_interactions["hint_count"]).clip(1, 5)

    # 2. Compute empirical problem difficulty = 1 - mean(correctness)
    prob_difficulty = 1.0 - df_interactions.groupby("problem_id")["correctness"].mean()
    prob_difficulty_dict = prob_difficulty.to_dict()

    # 3. Build problem -> (concept_id, concept_name, skill_name, question_type) mapping
    # Group skills by problem_id taking primary skill
    skills_primary = df_skills.drop_duplicates(subset=["problem_id"]).copy()
    concept_mappings = {}
    for _, row in skills_primary.iterrows():
        pid = row["problem_id"]
        cid, cname = map_node_code_to_concept(str(row.get("node_code", "")))
        concept_mappings[pid] = {
            "concept_id": cid,
            "concept_name": cname,
            "skill_name": str(row.get("node_name", cname)),
        }

    problem_type_dict = {}
    if "Problem Type" in df_problems.columns:
        for _, row in df_problems.drop_duplicates(subset=["problem_id"]).iterrows():
            pid = row["problem_id"]
            problem_type_dict[pid] = map_problem_type(row.get("Problem Type", ""))

    # 4. Filter and select learners
    user_counts = df_interactions.groupby("user_id").size()
    eligible_users = user_counts[user_counts >= min_interactions_per_learner].index.tolist()
    print(f"Found {len(eligible_users):,} learners with >= {min_interactions_per_learner} interactions.")

    if max_learners and len(eligible_users) > max_learners:
        # Deterministic sample
        rng = np.random.RandomState(42)
        selected_users = set(rng.choice(eligible_users, size=max_learners, replace=False))
    else:
        selected_users = set(eligible_users)

    print(f"Processing trajectories for {len(selected_users):,} learners...")
    df_filtered = df_interactions[df_interactions["user_id"].isin(selected_users)].copy()

    # Parse timestamps for response times & temporal gap
    df_filtered["end_time_dt"] = pd.to_datetime(df_filtered["end_time"], errors="coerce")
    df_filtered = df_filtered.sort_values(by=["user_id", "end_time_dt", "id"])

    # 5. Build interaction records
    records = []
    seen_users = set()

    for user_id, group in df_filtered.groupby("user_id"):
        # Truncate sequence to max_seq_len if specified
        seq = group.head(max_seq_len)
        seen_users.add(user_id)

        prev_time = None
        for idx, (_, row) in enumerate(seq.iterrows()):
            pid = row["problem_id"]
            curr_time = row["end_time_dt"]

            # Compute response time & temporal gap
            if prev_time is not None and pd.notnull(curr_time) and pd.notnull(prev_time):
                dt_sec = max(1.0, (curr_time - prev_time).total_seconds())
                response_time_sec = min(600.0, dt_sec)
                temporal_gap_hours = min(24.0, dt_sec / 3600.0)
            else:
                response_time_sec = 30.0  # default prior
                temporal_gap_hours = 0.5

            prev_time = curr_time

            cmap = concept_mappings.get(pid, {
                "concept_id": 0,
                "concept_name": CONCEPT_NAMES[0],
                "skill_name": "General Math Problem",
            })
            qtype = problem_type_dict.get(pid, "multiple_choice")
            difficulty = prob_difficulty_dict.get(pid, 0.5)

            records.append({
                "learner_id": str(user_id),
                "interaction_id": idx,
                "concept_id": cmap["concept_id"],
                "concept_name": cmap["concept_name"],
                "skill_name": cmap["skill_name"],
                "correctness": float(row["correctness"]),
                "difficulty": round(float(difficulty), 4),
                "response_time_sec": round(float(response_time_sec), 1),
                "attempts": int(row["attempts"]),
                "temporal_gap_hours": round(float(temporal_gap_hours), 2),
                "evaluator_confidence": 0.95,  # Verified real system log
                "question_type": qtype,
                "sim_hour": round(idx * 0.5, 1),
            })

    print(f"Total processed interaction records: {len(records):,} across {len(seen_users):,} learners.")

    # 6. Split 70% train / 15% val / 15% test by learner
    learner_list = sorted(list(seen_users))
    rng = np.random.RandomState(42)
    rng.shuffle(learner_list)

    n_tot = len(learner_list)
    n_test = int(0.15 * n_tot)
    n_val = int(0.15 * n_tot)

    test_ids = set(learner_list[:n_test])
    val_ids = set(learner_list[n_test:n_test + n_val])
    train_ids = set(learner_list[n_test + n_val:])

    train_recs = [r for r in records if r["learner_id"] in train_ids]
    val_recs = [r for r in records if r["learner_id"] in val_ids]
    test_recs = [r for r in records if r["learner_id"] in test_ids]

    print(f"Split sizes: Train={len(train_recs):,} ({len(train_ids)} learners), "
          f"Val={len(val_recs):,} ({len(val_ids)} learners), "
          f"Test={len(test_recs):,} ({len(test_ids)} learners).")

    # Save to data/foundational_assist/
    pd.DataFrame(train_recs).to_csv(output_dir / "train.csv", index=False)
    pd.DataFrame(val_recs).to_csv(output_dir / "val.csv", index=False)
    pd.DataFrame(test_recs).to_csv(output_dir / "test.csv", index=False)

    # 7. Update Layer 1 (data/layer1_public_kt/) so the repo baseline uses REAL data!
    layer1_dir = PROJECT_ROOT / "data" / "layer1_public_kt"
    layer1_dir.mkdir(parents=True, exist_ok=True)
    layer1_processed_path = layer1_dir / "assistments09_processed.csv"
    layer1_meta_path = layer1_dir / "layer1_metadata.json"

    # Save standardized Layer 1 format
    df_layer1 = pd.DataFrame(records)[[
        "learner_id", "skill_name", "correctness", "attempts", "response_time_sec", "interaction_id"
    ]].rename(columns={"interaction_id": "interaction_index"})

    with open(layer1_processed_path, "w", newline="") as f:
        f.write("# Layer 1 (real public data) — ASSISTments/FoundationalASSIST\n")
        f.write(f"# Source: Hugging Face ASSISTments/FoundationalASSIST | {len(records):,} rows | {len(seen_users):,} learners\n")
        f.write("# License: CC-BY-NC-4.0; ASSISTments Foundation\n")
        df_layer1.to_csv(f, index=False)

    metadata = {
        "source": "ASSISTments/FoundationalASSIST (Real Human Student Data)",
        "license": "CC-BY-NC-4.0",
        "huggingface_dataset": "ASSISTments/FoundationalASSIST",
        "n_rows": len(records),
        "n_learners": len(seen_users),
        "n_skills": len(set(r["skill_name"] for r in records)),
        "is_real_data": True,
        "processed_timestamp": datetime.now().isoformat(),
    }
    with open(layer1_meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSuccessfully generated FoundationalASSIST dataset in {output_dir}")
    print(f"Updated Layer 1 metadata at {layer1_meta_path} (is_real_data: True)")
    return train_recs, val_recs, test_recs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process FoundationalASSIST dataset")
    parser.add_argument("--max-learners", type=int, default=2500, help="Max learners to process")
    parser.add_argument("--max-seq-len", type=int, default=50, help="Max sequence length per learner")
    args = parser.parse_args()

    process_dataset(max_learners=args.max_learners, max_seq_len=args.max_seq_len)
