"""
Curriculum priority function and action policy for the UMiKT-GAT system.

Priority function (Section 9 of spec):
  P_i = α*(1-M_i) + β*MC_i + γ*U_i + δ*Dep_i + ε*F_i + ζ*GoalRelevance_i

Where:
  M_i  = mastery estimate for concept i ∈ [0,1]
  MC_i = misconception severity for concept i ∈ [0,1]
  U_i  = uncertainty for concept i ∈ [0,1] (from MC-Dropout variance)
  Dep_i = prerequisite dependency score ∈ [0,1] (from GAT propagation)
  F_i  = forgetting risk = 1 - R_i ∈ [0,1]
  GoalRelevance_i = learner's stated goal relevance for concept i ∈ [0,1]

Default coefficients:
  α=0.35, β=0.25, γ=0.15, δ=0.10, ε=0.10, ζ=0.05

IMPORTANT: These coefficients are HEURISTIC weights, NOT a novel contribution.
They are domain-expert priors used as a baseline priority function.
The paper must state explicitly:
  "The priority weights α..ζ are manually set heuristic values used as a
   baseline interpretable policy. Learning these weights (e.g. via contextual
   bandit or multi-objective optimization) is left for future work."

Action policy: Explicit if/else rules over the four-part learner state.
The policy is interpretable by design. It matches the policy table in the spec.
"""


N_CONCEPTS = 7

# Heuristic priority weights (not novel — documented as baseline)
DEFAULT_PRIORITY_WEIGHTS = {
    "alpha": 0.35,   # mastery gap weight
    "beta": 0.25,    # misconception weight
    "gamma": 0.15,   # uncertainty weight
    "delta": 0.10,   # prerequisite dependency weight
    "epsilon": 0.10, # forgetting risk weight
    "zeta": 0.05,    # goal relevance weight
}

# Action labels matching the spec
ACTIONS = [
    "advance_to_next_topic",
    "practice_current_concept",
    "misconception_targeted_remediation",
    "diagnose_prerequisite",
    "adaptive_diagnostic_assessment",
    "schedule_spaced_revision",
    "misconception_targeted_content",
]

# Concept names for explanations
CONCEPT_NAMES = [
    "Variables", "Conditions", "Loops", "Functions",
    "Recursion", "Data Structures", "Algorithms"
]

# Prerequisite map for dependency scoring (matches concept_graph.json)
PREREQUISITES = {
    0: [],      # Variables: no prereqs
    1: [0],     # Conditions: Variables
    2: [0, 1],  # Loops: Variables, Conditions
    3: [2],     # Functions: Loops
    4: [3, 1],  # Recursion: Functions, Conditions
    5: [3, 2],  # Data Structures: Functions, Loops
    6: [5],     # Algorithms: Data Structures
}


class CurriculumPriority:
    """
    Computes per-concept priority scores for curriculum selection.

    Priority P_i determines which concept the system should focus on next.
    Higher priority → more urgent for attention.

    Args:
        weights: Dict of priority weights. Defaults to DEFAULT_PRIORITY_WEIGHTS.
    """

    def __init__(self, weights: dict | None = None):
        self.weights = weights or DEFAULT_PRIORITY_WEIGHTS.copy()

    def compute_dependency_scores(
        self,
        mastery: list[float],
    ) -> list[float]:
        """
        Compute prerequisite dependency score for each concept.

        Dep_i = average(1 - mastery[prereq]) over prerequisites of i.
        High Dep_i means prerequisite weaknesses may be causing low mastery in i.

        Args:
            mastery: Mastery estimates for all concepts.

        Returns:
            Dependency scores Dep_i ∈ [0,1] for each concept.
        """
        dep_scores = []
        for c in range(N_CONCEPTS):
            prereqs = PREREQUISITES.get(c, [])
            if not prereqs:
                dep_scores.append(0.0)
            else:
                avg_prereq_gap = sum(1.0 - mastery[p] for p in prereqs) / len(prereqs)
                dep_scores.append(avg_prereq_gap)
        return dep_scores

    def compute_priorities(
        self,
        mastery: list[float],
        misconception_severity: list[float],
        uncertainty: list[float],
        forgetting_risk: list[float],
        goal_relevance: list[float] | None = None,
    ) -> list[float]:
        """
        Compute priority P_i for each concept.

        Args:
            mastery: M_i ∈ [0,1] per concept.
            misconception_severity: MC_i ∈ [0,1] per concept.
            uncertainty: U_i ∈ [0,1] per concept (normalized variance).
            forgetting_risk: F_i = 1 - R_i ∈ [0,1] per concept.
            goal_relevance: GoalRelevance_i ∈ [0,1]. Default: uniform 1.0.

        Returns:
            Priority scores P_i ∈ [0,1] for each concept.
        """
        if goal_relevance is None:
            goal_relevance = [1.0] * N_CONCEPTS

        dep_scores = self.compute_dependency_scores(mastery)

        w = self.weights
        priorities = []
        for i in range(N_CONCEPTS):
            gap_i = 1.0 - mastery[i]
            p_i = (
                w["alpha"] * gap_i
                + w["beta"] * misconception_severity[i]
                + w["gamma"] * uncertainty[i]
                + w["delta"] * dep_scores[i]
                + w["epsilon"] * forgetting_risk[i]
                + w["zeta"] * goal_relevance[i]
            )
            priorities.append(p_i)

        return priorities

    def top_concepts(
        self,
        priorities: list[float],
        n: int = 3,
    ) -> list[tuple[int, float]]:
        """
        Return top-n concepts by priority.

        Returns:
            List of (concept_id, priority) tuples, sorted descending.
        """
        ranked = sorted(
            [(i, priorities[i]) for i in range(N_CONCEPTS)],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:n]


class ActionPolicy:
    """
    Interpretable if/else action selection policy over the learner state.

    Selects next curriculum action based on the four-part learner state
    for the highest-priority concept. All rules are explicit and traceable.

    Thresholds are heuristic and documented as such — not novel contributions.
    """

    # Threshold constants (heuristic, documented)
    HIGH_MASTERY = 0.75
    LOW_MASTERY = 0.40
    HIGH_MISCONCEPTION = 0.50
    HIGH_UNCERTAINTY = 0.40
    HIGH_FORGETTING = 0.40
    HIGH_PREREQUISITE_GAP = 0.50

    def select_action(
        self,
        concept_id: int,
        mastery: float,
        misconception_severity: float,
        uncertainty: float,
        forgetting_risk: float,
        dep_score: float,
    ) -> tuple[str, str]:
        """
        Select curriculum action for the given concept and learner state.

        Returns:
            Tuple of (action_label, explanation_string).
            The explanation string is human-readable for the learner/tutor.
        """
        cname = CONCEPT_NAMES[concept_id] if concept_id < N_CONCEPTS else f"Concept {concept_id}"

        # ── Rule 1: High mastery + low uncertainty → Advance ─────────────────
        if (mastery >= self.HIGH_MASTERY
                and uncertainty < self.HIGH_UNCERTAINTY
                and misconception_severity < self.HIGH_MISCONCEPTION
                and forgetting_risk < self.HIGH_FORGETTING):
            return (
                "advance_to_next_topic",
                f"{cname} shows high mastery (M={mastery:.2f}) with low uncertainty "
                f"(U={uncertainty:.2f}) and no significant misconception or forgetting risk. "
                f"Ready to advance to the next topic."
            )

        # ── Rule 2: High mastery + high forgetting → Spaced revision ─────────
        if mastery >= self.HIGH_MASTERY and forgetting_risk >= self.HIGH_FORGETTING:
            return (
                "schedule_spaced_revision",
                f"{cname} was previously mastered (M={mastery:.2f}) but shows elevated "
                f"forgetting risk (F={forgetting_risk:.2f}). Scheduling spaced revision "
                f"to consolidate long-term retention."
            )

        # ── Rule 3: Low mastery + high misconception → Misconception content ──
        if (mastery < self.LOW_MASTERY + 0.2
                and misconception_severity >= self.HIGH_MISCONCEPTION):
            return (
                "misconception_targeted_content",
                f"{cname} has moderate-to-low mastery (M={mastery:.2f}) and a persistent "
                f"misconception (MC severity={misconception_severity:.2f}). Providing "
                f"misconception-targeted instructional content to address the root confusion."
            )

        # ── Rule 4: Low mastery + high prerequisite gap → Diagnose prereq ────
        if mastery < self.LOW_MASTERY and dep_score >= self.HIGH_PREREQUISITE_GAP:
            prereqs = PREREQUISITES.get(concept_id, [])
            prereq_names = [CONCEPT_NAMES[p] for p in prereqs if p < N_CONCEPTS]
            return (
                "diagnose_prerequisite",
                f"{cname} shows low mastery (M={mastery:.2f}) and weak prerequisites "
                f"(candidate gaps in: {', '.join(prereq_names) or 'none'}). "
                f"Diagnosing prerequisite knowledge before attempting {cname} content."
            )

        # ── Rule 5: Moderate mastery + high uncertainty → Diagnostic assessment
        if uncertainty >= self.HIGH_UNCERTAINTY:
            return (
                "adaptive_diagnostic_assessment",
                f"{cname} has uncertain mastery estimate (U={uncertainty:.2f}, "
                f"M={mastery:.2f}). Running additional diagnostic assessment to "
                f"gather more evidence before making a curriculum decision."
            )

        # ── Rule 6: Low mastery, no special signal → Remediate ───────────────
        if mastery < self.LOW_MASTERY:
            return (
                "misconception_targeted_remediation",
                f"{cname} shows low mastery (M={mastery:.2f}) without specific "
                f"misconception or high prerequisite gap. Providing targeted remediation "
                f"and additional practice."
            )

        # ── Rule 7: Default → Practice ────────────────────────────────────────
        return (
            "practice_current_concept",
            f"{cname} is in progress (M={mastery:.2f}). Continuing practice to "
            f"build toward mastery threshold (≥{self.HIGH_MASTERY:.0%})."
        )
