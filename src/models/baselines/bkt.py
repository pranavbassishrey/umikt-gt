"""
Bayesian Knowledge Tracing (BKT) baseline implementation.

BKT is a 4-parameter Hidden Markov Model (HMM) for knowledge tracing.
It is a standard baseline in KT literature (Corbett & Anderson 1994).

Parameters per skill:
  p_learn (L0):  Probability of knowing the skill initially
  p_transit (T): Probability of transitioning from unknown to known
  p_slip (S):    Probability of answering wrong despite knowing (slip)
  p_guess (G):   Probability of answering correctly despite not knowing (guess)

This is a simplified reimplementation, not a novel contribution.
It serves as a baseline comparison point for mastery prediction.

Fitting: Per-skill EM algorithm (Baum-Welch) for parameter estimation.
Inference: Forward algorithm for P(mastery | observations).

Limitations of this reimplementation:
  - EM may converge to local optima (standard BKT limitation).
  - No skill chaining or prerequisite modeling (by design — this is a baseline).
  - Treats each skill independently.
  - Using synthetic data means baseline comparison is on synthetic targets.
  
These limitations are documented in RESULTS_SUMMARY.md and DATASET_CARD.md.
"""

from collections import defaultdict
import math


class BKT:
    """
    Per-skill BKT model fitted via EM (Baum-Welch).

    Args:
        n_em_iters: Number of EM iterations.
        p_learn_init: Initial p_learn estimate.
        p_transit_init: Initial p_transit estimate.
        p_slip_init: Initial p_slip estimate.
        p_guess_init: Initial p_guess estimate.
    """

    def __init__(
        self,
        n_em_iters: int = 50,
        p_learn_init: float = 0.3,
        p_transit_init: float = 0.1,
        p_slip_init: float = 0.1,
        p_guess_init: float = 0.2,
    ):
        self.n_em_iters = n_em_iters
        self.params_init = {
            "p_learn": p_learn_init,
            "p_transit": p_transit_init,
            "p_slip": p_slip_init,
            "p_guess": p_guess_init,
        }
        # skill_name → fitted params
        self.skill_params: dict[str, dict] = {}

    def _clip(self, x: float, lo: float = 0.01, hi: float = 0.99) -> float:
        return max(lo, min(hi, x))

    def _emission_prob(self, p_know: float, correct: int, p_slip: float, p_guess: float) -> float:
        """P(observation | state)"""
        if correct:
            return p_know * (1 - p_slip) + (1 - p_know) * p_guess
        else:
            return p_know * p_slip + (1 - p_know) * (1 - p_guess)

    def _forward_algorithm(
        self,
        sequence: list[int],
        p_learn: float,
        p_transit: float,
        p_slip: float,
        p_guess: float,
    ) -> list[float]:
        """
        Run forward algorithm to compute P(mastered | observations up to t).

        Returns:
            List of P(mastered | obs_1..t) for each timestep t.
        """
        p_know = p_learn
        posteriors = []

        for obs in sequence:
            # Emission probability
            p_obs = self._emission_prob(p_know, obs, p_slip, p_guess)
            if p_obs < 1e-10:
                p_obs = 1e-10

            # Update: P(know | obs) via Bayes
            p_know_given_obs = (p_know * (1 - p_slip if obs else p_slip)) / p_obs
            p_not_know_given_obs = ((1 - p_know) * (p_guess if obs else 1 - p_guess)) / p_obs
            p_know_given_obs = self._clip(p_know_given_obs)

            # Normalize
            total = p_know_given_obs + p_not_know_given_obs
            if total > 0:
                p_know_given_obs /= total

            posteriors.append(p_know_given_obs)

            # Transition to next timestep
            p_know = p_know_given_obs + (1 - p_know_given_obs) * p_transit

        return posteriors

    def _em_step(
        self,
        sequences: list[list[int]],
        params: dict,
    ) -> dict:
        """One E-M step for a single skill."""
        p_learn = params["p_learn"]
        p_transit = params["p_transit"]
        p_slip = params["p_slip"]
        p_guess = params["p_guess"]

        sum_pknow_0 = 0.0
        n_seqs = 0
        sum_trans_num = 0.0
        sum_trans_den = 0.0
        sum_slip_num = 0.0
        sum_guess_num = 0.0
        sum_know_obs_correct = 0.0
        sum_know_obs = 0.0
        sum_not_know_obs_correct = 0.0
        sum_not_know_obs = 0.0

        for seq in sequences:
            if not seq:
                continue
            posteriors = self._forward_algorithm(seq, p_learn, p_transit, p_slip, p_guess)

            # Accumulate sufficient statistics
            sum_pknow_0 += posteriors[0]
            n_seqs += 1

            p_k = p_learn
            for t, obs in enumerate(seq):
                p_ko = posteriors[t]
                p_nko = 1 - p_ko

                # Transition statistics (up to T-1)
                if t < len(seq) - 1:
                    sum_trans_num += p_nko * p_transit
                    sum_trans_den += p_nko

                # Slip/guess statistics
                if obs:
                    sum_know_obs_correct += p_ko
                    sum_not_know_obs_correct += p_nko
                sum_know_obs += p_ko
                sum_not_know_obs += p_nko

        # M-step: update parameters
        new_p_learn = self._clip(sum_pknow_0 / max(n_seqs, 1))
        new_p_transit = self._clip(sum_trans_num / max(sum_trans_den, 1e-10))
        new_p_slip = self._clip(1 - sum_know_obs_correct / max(sum_know_obs, 1e-10))
        new_p_guess = self._clip(sum_not_know_obs_correct / max(sum_not_know_obs, 1e-10))

        return {
            "p_learn": new_p_learn,
            "p_transit": new_p_transit,
            "p_slip": new_p_slip,
            "p_guess": new_p_guess,
        }

    def fit(self, records: list[dict], skill_col: str = "concept_id"):
        """
        Fit BKT parameters per skill/concept using EM.

        Args:
            records: List of interaction dicts with 'learner_id', skill_col, 'correctness'.
            skill_col: Column name for skill/concept identifier.
        """
        # Group sequences by (skill, learner)
        skill_sequences: dict = defaultdict(lambda: defaultdict(list))
        for r in records:
            skill = str(r.get(skill_col, "unknown"))
            learner = r.get("learner_id", "unknown")
            correct = int(float(r.get("correctness", 0)))
            skill_sequences[skill][learner].append(correct)

        # Fit each skill independently
        for skill, learner_seqs in skill_sequences.items():
            seqs = list(learner_seqs.values())
            params = self.params_init.copy()

            for _ in range(self.n_em_iters):
                new_params = self._em_step(seqs, params)
                # Check convergence
                delta = max(
                    abs(new_params[k] - params[k]) for k in params
                )
                params = new_params
                if delta < 1e-6:
                    break

            self.skill_params[skill] = params

    def predict_sequence(self, skill: str, sequence: list[int]) -> list[float]:
        """
        Predict mastery probability at each timestep for a given skill.

        Args:
            skill: Skill/concept identifier.
            sequence: List of binary correctness observations.

        Returns:
            List of mastery probabilities.
        """
        if skill not in self.skill_params:
            # Unknown skill: use init params
            params = self.params_init
        else:
            params = self.skill_params[skill]

        return self._forward_algorithm(
            sequence,
            params["p_learn"],
            params["p_transit"],
            params["p_slip"],
            params["p_guess"],
        )

    def predict_all(self, records: list[dict], skill_col: str = "concept_id") -> list[float]:
        """
        Predict mastery for all records.

        Returns:
            List of predicted mastery probabilities (same order as records).
        """
        # Rebuild sequences from records, predict, flatten
        from collections import OrderedDict
        seq_by_skill_learner: dict = defaultdict(list)
        pred_idx = defaultdict(int)

        # First pass: build sequences
        learner_skill_seqs = defaultdict(lambda: defaultdict(list))
        for r in records:
            skill = str(r.get(skill_col, "unknown"))
            learner = r.get("learner_id", "unknown")
            correct = int(float(r.get("correctness", 0)))
            learner_skill_seqs[learner][skill].append(correct)

        # Second pass: for each record, compute mastery from sequence up to that point
        # This requires knowing the index within the sequence — maintained via counter
        counters = defaultdict(lambda: defaultdict(int))
        predictions = []

        for r in records:
            skill = str(r.get(skill_col, "unknown"))
            learner = r.get("learner_id", "unknown")
            idx = counters[learner][skill]
            counters[learner][skill] += 1

            seq_so_far = learner_skill_seqs[learner][skill][:idx + 1]
            posteriors = self.predict_sequence(skill, seq_so_far)
            predictions.append(posteriors[-1] if posteriors else 0.3)

        return predictions
