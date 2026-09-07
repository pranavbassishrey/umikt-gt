"""
Synthetic learner trajectory generator for the UMiKT-GAT project.

PURPOSE: Generate clearly-labeled synthetic training/evaluation data for the
Programming Fundamentals domain. This data is used to:
  1. Validate that the UMiKT-GAT model can recover known latent states.
  2. Enable end-to-end pipeline evaluation when real misconception labels are unavailable.
  3. Provide a controlled benchmark for ablation studies.

DATA LABELING REQUIREMENT: All outputs of this script are SYNTHETIC. Every
generated file includes the string "(synthetic)" in its name and content headers.
Never present these outputs as real human subject data.

Simulation assumptions (explicit, not hidden):
- Ground-truth mastery follows a logistic growth model with individual learning rates.
- Ground-truth misconception state follows a first-order Markov chain.
- Ground-truth forgetting follows exponential decay between sessions.
- Observed correctness is a noisy Bernoulli sample from (mastery - misconception_penalty).
- Response times are sampled from concept-difficulty-conditioned log-normal distributions.

These are modeling assumptions, not empirical findings. Their role is to create
a reproducible synthetic environment where the true latent state is known.
"""

import json
import random
import math
import csv
import os
import sys
from pathlib import Path

# ─────────────────────────────────────────────
# Constants (all explicit, justified in comments)
# ─────────────────────────────────────────────

CONCEPT_NAMES = [
    "Variables", "Conditions", "Loops",
    "Functions", "Recursion", "Data Structures", "Algorithms"
]
N_CONCEPTS = len(CONCEPT_NAMES)

# Misconception class IDs matching misconception_taxonomy.md
MISCONCEPTION_CLASSES = [
    "none",
    "syntax_misunderstanding",
    "variable_scope_misunderstanding",
    "loop_termination_misunderstanding",
    "function_parameter_misunderstanding",
    "recursion_base_case_misunderstanding",
    "reference_vs_value_misunderstanding",
]
N_MISCONCEPTION_CLASSES = len(MISCONCEPTION_CLASSES)

# Concept difficulty (0=easy, 1=hard) — used to scale response times
# These are domain-expert priors, not learned parameters.
CONCEPT_DIFFICULTY = [0.2, 0.3, 0.45, 0.55, 0.80, 0.65, 0.75]

# Affinity: which misconception classes are most likely per concept
# Row = concept_id, columns = probability over misconception classes
# Constructed to reflect the misconception-concept affinity in taxonomy doc.
CONCEPT_MISCONCEPTION_AFFINITY = [
    # Vars:   none  syntax  scope  loop   param  recurs refval
    [0.65, 0.20, 0.10, 0.01, 0.01, 0.01, 0.02],
    # Cond:
    [0.60, 0.25, 0.08, 0.03, 0.02, 0.01, 0.01],
    # Loop:
    [0.55, 0.12, 0.08, 0.18, 0.03, 0.02, 0.02],
    # Func:
    [0.55, 0.05, 0.15, 0.02, 0.18, 0.02, 0.03],
    # Recur:
    [0.50, 0.03, 0.05, 0.04, 0.10, 0.25, 0.03],
    # DS:
    [0.55, 0.02, 0.03, 0.08, 0.02, 0.02, 0.28],
    # Algo:
    [0.52, 0.02, 0.04, 0.15, 0.02, 0.10, 0.15],
]

# Misconception severity given class is non-none: Beta(2,3) approximated
# A misconception, when present, is more likely to be mild-moderate than severe.
MISCONCEPTION_SEVERITY_MEAN = 0.40
MISCONCEPTION_SEVERITY_STD = 0.20

# Forgetting rate: global λ prior
FORGETTING_RATE_MEAN = 0.05   # per hour
FORGETTING_RATE_STD = 0.02    # learner variation

# Learning rate (logistic growth rate) prior
LEARNING_RATE_MEAN = 0.15
LEARNING_RATE_STD = 0.05


def _softmax(x):
    m = max(x)
    e = [math.exp(v - m) for v in x]
    s = sum(e)
    return [v / s for v in e]


def _sample_categorical(probs, rng):
    """Sample from a categorical distribution given un-normalized probabilities."""
    s = sum(probs)
    normalized = [p / s for p in probs]
    u = rng.random()
    cum = 0.0
    for i, p in enumerate(normalized):
        cum += p
        if u <= cum:
            return i
    return len(probs) - 1


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _sample_normal_positive(mean, std, rng):
    """Sample from a truncated normal (clipped to positive)."""
    return max(0.01, rng.gauss(mean, std))


class LearnerSimulator:
    """
    Simulates a single learner's trajectory through the Programming Fundamentals domain.

    Ground truth:
      - mastery[c] ∈ [0,1]: true knowledge mastery for concept c
      - misconception_class[c]: current dominant misconception class for concept c
      - misconception_severity[c] ∈ [0,1]: severity of current misconception
      - forgetting_rate[c]: λ parameter for exponential forgetting
      - last_practice_hour[c]: simulated timestamp of last meaningful practice

    These ground-truth states are stored in the generated data and used for
    supervised evaluation (model should recover them). They are NEVER presented
    as real student measurements.
    """

    def __init__(self, learner_id, rng):
        self.learner_id = learner_id
        self.rng = rng

        # Sample individual learning rates per concept
        self.learning_rates = [
            _sample_normal_positive(LEARNING_RATE_MEAN, LEARNING_RATE_STD, rng)
            for _ in range(N_CONCEPTS)
        ]

        # Sample individual forgetting rates per concept
        self.forgetting_rates = [
            _sample_normal_positive(FORGETTING_RATE_MEAN, FORGETTING_RATE_STD, rng)
            for _ in range(N_CONCEPTS)
        ]

        # Initial mastery: low, scaled by concept position (later concepts harder to start)
        self.mastery = [
            _clip(rng.gauss(0.10 - i * 0.01, 0.05))
            for i in range(N_CONCEPTS)
        ]

        # Initial misconception state
        self.misconception_class = [
            _sample_categorical(CONCEPT_MISCONCEPTION_AFFINITY[c], rng)
            for c in range(N_CONCEPTS)
        ]

        self.misconception_severity = [
            _clip(rng.gauss(MISCONCEPTION_SEVERITY_MEAN, MISCONCEPTION_SEVERITY_STD))
            if self.misconception_class[c] != 0 else 0.0
            for c in range(N_CONCEPTS)
        ]

        # Simulated time tracking (hours from start of session)
        self.last_practice_hour = [0.0] * N_CONCEPTS
        self.current_hour = 0.0

        # Interaction count per concept
        self.interaction_count = [0] * N_CONCEPTS

    def _apply_forgetting(self, concept_id):
        """Apply exponential forgetting based on time since last practice."""
        delta_t = self.current_hour - self.last_practice_hour[concept_id]
        if delta_t > 0:
            self.mastery[concept_id] *= math.exp(
                -self.forgetting_rates[concept_id] * delta_t
            )
            self.mastery[concept_id] = _clip(self.mastery[concept_id])

    def _compute_prerequisite_boost(self, concept_id):
        """
        Prerequisite mastery provides a soft boost to the learner's effective performance.
        This is a simulation assumption, not a modeled output.
        """
        # Hardcoded prerequisite relationships matching concept_graph.json
        prereqs = {
            1: [0], 2: [0, 1], 3: [2], 4: [3, 1],
            5: [3, 2], 6: [5]
        }
        deps = prereqs.get(concept_id, [])
        if not deps:
            return 0.0
        avg_prereq = sum(self.mastery[d] for d in deps) / len(deps)
        return 0.15 * avg_prereq  # soft boost, capped at 0.15

    def _update_mastery(self, concept_id, correctness):
        """
        Update mastery using a simplified logistic growth model.
        Correct answers increase mastery; misconceptions dampen the increase.
        """
        prereq_boost = self._compute_prerequisite_boost(concept_id)
        current = self.mastery[concept_id]

        if correctness > 0.5:
            # Correct: logistic-style increment, larger when far from ceiling
            delta = self.learning_rates[concept_id] * (1.0 - current) * (1 + prereq_boost)
            # Misconception slows learning even when correct (student may be pattern-matching)
            mc_penalty = 0.3 * self.misconception_severity[concept_id]
            self.mastery[concept_id] = _clip(current + delta * (1 - mc_penalty))
        else:
            # Incorrect: slight decrease
            self.mastery[concept_id] = _clip(current - 0.02)

        # Misconception resolution: small probability of resolving on correct answer
        if correctness > 0.5 and self.misconception_class[concept_id] != 0:
            resolve_prob = 0.15 * correctness
            if self.rng.random() < resolve_prob:
                self.misconception_class[concept_id] = 0
                self.misconception_severity[concept_id] = 0.0

    def _sample_observed_correctness(self, concept_id):
        """
        Sample a noisy observed correctness given ground-truth mastery and misconception.
        Misconceptions reduce effective probability of correct answer.
        """
        p_correct = self.mastery[concept_id]
        mc_penalty = 0.4 * self.misconception_severity[concept_id]
        p_correct = _clip(p_correct - mc_penalty)

        # Bernoulli sample with small slip/guess added
        p_slip = 0.05
        p_guess = 0.08
        p_obs = (1 - p_slip) * p_correct + p_guess * (1 - p_correct)
        return 1.0 if self.rng.random() < p_obs else 0.0

    def _sample_response_time(self, concept_id, correctness):
        """
        Sample a response time in seconds from a log-normal distribution.
        Harder concepts → longer times. Incorrect answers → longer times.
        """
        difficulty = CONCEPT_DIFFICULTY[concept_id]
        mu_log = math.log(30 + 90 * difficulty)  # base: 30-120 seconds
        sigma_log = 0.4
        if correctness < 0.5:
            mu_log += 0.3  # incorrect answers take longer
        t = math.exp(self.rng.gauss(mu_log, sigma_log))
        return round(_clip(t, 5.0, 600.0), 1)

    def _pick_concept_to_practice(self):
        """Simple curriculum: pick concept with lowest mastery that has prerequisites met."""
        # Prerequisite thresholds
        prereqs = {
            1: [0], 2: [0, 1], 3: [2], 4: [3, 1],
            5: [3, 2], 6: [5]
        }
        eligible = []
        for c in range(N_CONCEPTS):
            deps = prereqs.get(c, [])
            if all(self.mastery[d] >= 0.3 for d in deps):
                eligible.append(c)

        if not eligible:
            eligible = [0]

        # Weight by (1 - mastery) + misconception severity
        weights = [
            (1 - self.mastery[c]) + 0.3 * self.misconception_severity[c]
            for c in eligible
        ]
        return eligible[_sample_categorical(weights, self.rng)]

    def generate_interaction(self, concept_id=None):
        """
        Simulate one learner-concept interaction and return an interaction record.

        Returns a dict representing one row of the dataset.
        All ground-truth fields are included for evaluation purposes.
        """
        # Advance simulated time (sessions spaced 0.5-4 hours apart)
        time_gap = self.rng.uniform(0.5, 4.0)
        self.current_hour += time_gap

        if concept_id is None:
            concept_id = self._pick_concept_to_practice()

        # Apply forgetting before this interaction
        self._apply_forgetting(concept_id)

        # Sample observed correctness
        correctness = self._sample_observed_correctness(concept_id)

        # Sample response time
        response_time = self._sample_response_time(concept_id, correctness)

        # Attempts (1-3)
        attempts = 1
        if correctness < 0.5 and self.rng.random() < 0.4:
            attempts = self.rng.randint(2, 3)

        # Evaluator confidence (simulated LLM extraction confidence)
        evaluator_confidence = _clip(self.rng.gauss(0.75, 0.15))

        # Question type
        question_types = ["multiple_choice", "short_answer", "code_completion", "debugging"]
        question_type = self.rng.choice(question_types)

        # Record interaction
        record = {
            # Identifiers
            "learner_id": self.learner_id,
            "interaction_id": sum(self.interaction_count),
            "concept_id": concept_id,
            "concept_name": CONCEPT_NAMES[concept_id],
            # Observed features
            "correctness": correctness,
            "difficulty": CONCEPT_DIFFICULTY[concept_id],
            "response_time_sec": response_time,
            "attempts": attempts,
            "question_type": question_type,
            "temporal_gap_hours": round(time_gap, 2),
            "evaluator_confidence": round(evaluator_confidence, 4),
            # Ground-truth labels (for supervised evaluation)
            "gt_mastery": round(self.mastery[concept_id], 4),
            "gt_misconception_class": self.misconception_class[concept_id],
            "gt_misconception_label": MISCONCEPTION_CLASSES[self.misconception_class[concept_id]],
            "gt_misconception_severity": round(self.misconception_severity[concept_id], 4),
            "gt_forgetting_rate": round(self.forgetting_rates[concept_id], 4),
            "gt_retention": round(
                self.mastery[concept_id] * math.exp(
                    -self.forgetting_rates[concept_id] * time_gap
                ), 4
            ),
            # Simulated time
            "sim_hour": round(self.current_hour, 2),
        }

        # Update ground-truth state after interaction
        self._update_mastery(concept_id, correctness)
        self.last_practice_hour[concept_id] = self.current_hour
        self.interaction_count[concept_id] += 1

        return record


def generate_dataset(
    n_learners: int,
    n_interactions_per_learner: int,
    seed: int,
    output_dir: Path,
    split_name: str,
):
    """
    Generate a synthetic dataset split.

    Args:
        n_learners: Number of simulated learners.
        n_interactions_per_learner: Interactions per learner.
        seed: Random seed for reproducibility.
        output_dir: Directory to write the CSV file.
        split_name: 'train', 'val', or 'test'.

    Returns:
        Path to the written CSV file.
    """
    rng = random.Random(seed)
    records = []

    for i in range(n_learners):
        learner_id = f"synthetic_learner_{i:04d}"
        sim = LearnerSimulator(learner_id, rng)
        for _ in range(n_interactions_per_learner):
            rec = sim.generate_interaction()
            records.append(rec)

    os.makedirs(output_dir, exist_ok=True)
    out_path = output_dir / f"synthetic_{split_name}.csv"

    fieldnames = list(records[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # ── SYNTHETIC DATA LABEL ── must appear in the file header ──
        f.write("# SYNTHETIC DATA — generated by generate_synthetic.py\n")
        f.write("# Do NOT present as real human subject data.\n")
        f.write(f"# Split: {split_name} | N_learners: {n_learners} "
                f"| N_interactions_per_learner: {n_interactions_per_learner} "
                f"| seed: {seed}\n")
        writer.writeheader()
        writer.writerows(records)

    return out_path


def main():
    """Generate train, val, and test splits for the synthetic dataset."""
    base_dir = Path(__file__).parent
    output_dir = base_dir

    print("=" * 60)
    print("UMiKT-GAT SYNTHETIC DATA GENERATOR")
    print("ALL DATA GENERATED HERE IS SYNTHETIC.")
    print("=" * 60)

    # Train: 200 learners × 30 interactions = 6,000 rows
    print("\nGenerating training split (synthetic)...")
    train_path = generate_dataset(
        n_learners=200,
        n_interactions_per_learner=30,
        seed=42,
        output_dir=output_dir,
        split_name="train",
    )
    print(f"  → Written: {train_path}")

    # Val: 50 learners × 30 interactions = 1,500 rows
    print("\nGenerating validation split (synthetic)...")
    val_path = generate_dataset(
        n_learners=50,
        n_interactions_per_learner=30,
        seed=123,
        output_dir=output_dir,
        split_name="val",
    )
    print(f"  → Written: {val_path}")

    # Test: 50 learners × 30 interactions = 1,500 rows
    print("\nGenerating test split (synthetic)...")
    test_path = generate_dataset(
        n_learners=50,
        n_interactions_per_learner=30,
        seed=999,
        output_dir=output_dir,
        split_name="test",
    )
    print(f"  → Written: {test_path}")

    # Write metadata JSON
    metadata = {
        "description": "(SYNTHETIC) UMiKT-GAT synthetic training data",
        "warning": "THIS IS SYNTHETIC DATA. Do not present as real human subject data.",
        "splits": {
            "train": {"n_learners": 200, "n_interactions": 30, "seed": 42, "total_rows": 6000},
            "val":   {"n_learners": 50,  "n_interactions": 30, "seed": 123, "total_rows": 1500},
            "test":  {"n_learners": 50,  "n_interactions": 30, "seed": 999, "total_rows": 1500},
        },
        "n_concepts": N_CONCEPTS,
        "concept_names": CONCEPT_NAMES,
        "misconception_classes": MISCONCEPTION_CLASSES,
        "features": [
            "correctness", "difficulty", "response_time_sec", "attempts",
            "concept_id", "question_type", "temporal_gap_hours", "evaluator_confidence"
        ],
        "ground_truth_labels": [
            "gt_mastery", "gt_misconception_class", "gt_misconception_label",
            "gt_misconception_severity", "gt_forgetting_rate", "gt_retention"
        ],
        "simulation_assumptions": [
            "Mastery: logistic growth with individual learning rates",
            "Misconceptions: first-order Markov chain over taxonomy",
            "Forgetting: exponential decay with concept-specific lambda",
            "Correctness: noisy Bernoulli sample from (mastery - mc_penalty)",
            "Response time: log-normal conditioned on difficulty"
        ]
    }
    with open(output_dir / "synthetic_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  → Metadata: {output_dir / 'synthetic_metadata.json'}")
    print("\nDone. Remember: all generated data is SYNTHETIC.")


if __name__ == "__main__":
    main()
