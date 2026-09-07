"""
All 7 UMiKT-GAT agents + Orchestrator.

Agents are plain Python classes wired together by the Orchestrator.
No heavy framework is used — pure Python async/sync calls are sufficient
for a demo pipeline of this scale.

Agent responsibilities:
  DiagnosisAgent      → initial assessment, learner profile init
  AssessmentAgent     → select/generate assessment items
  EvaluationAgent     → evaluate answers, call LLM extraction
  LearnerModelingAgent → run UMiKT-GAT inference, return S_i
  CurriculumPlanningAgent → generate learning path
  CurriculumAdaptationAgent → select next action via priority + policy
  SupervisorAgent     → orchestrate the closed loop, log everything
"""

import json
import random
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import torch
import numpy as np

from .base_agent import BaseAgent
from src.llm.schema import ExtractionRequest
from src.llm.mock_provider import MockLLMProvider
from src.curriculum.priority import CurriculumPriority, ActionPolicy, CONCEPT_NAMES, N_CONCEPTS
from src.features.feature_builder import build_feature_vector, INPUT_DIM


# ─── Learner State ────────────────────────────────────────────────────────────

@dataclass
class LearnerState:
    """Four-part learner state per concept: S_i = [M_i, MC_i, U_i, R_i]"""
    mastery: list[float] = field(default_factory=lambda: [0.3] * N_CONCEPTS)
    misconception_severity: list[float] = field(default_factory=lambda: [0.0] * N_CONCEPTS)
    misconception_class: list[int] = field(default_factory=lambda: [0] * N_CONCEPTS)
    uncertainty: list[float] = field(default_factory=lambda: [0.5] * N_CONCEPTS)
    retention: list[float] = field(default_factory=lambda: [0.8] * N_CONCEPTS)
    forgetting_risk: list[float] = field(default_factory=lambda: [0.2] * N_CONCEPTS)
    interaction_count: list[int] = field(default_factory=lambda: [0] * N_CONCEPTS)

    def to_tensor(self, device="cpu") -> torch.Tensor:
        """Return (N_CONCEPTS, 4) tensor: [M, MC_sev, U, R]"""
        rows = [
            [self.mastery[i], self.misconception_severity[i],
             self.uncertainty[i], self.retention[i]]
            for i in range(N_CONCEPTS)
        ]
        return torch.tensor(rows, dtype=torch.float32, device=device)

    def summary(self) -> dict:
        return {
            CONCEPT_NAMES[i]: {
                "M": round(self.mastery[i], 3),
                "MC_sev": round(self.misconception_severity[i], 3),
                "MC_class": self.misconception_class[i],
                "U": round(self.uncertainty[i], 3),
                "R": round(self.retention[i], 3),
                "F": round(self.forgetting_risk[i], 3),
            }
            for i in range(N_CONCEPTS)
        }


# ─── Question Bank (small, for demo) ─────────────────────────────────────────

QUESTION_BANK = {
    0: [  # Variables
        {"q": "What is the output of: x = 5; x = x + 2; print(x)?", "a": "7", "difficulty": 0.2},
        {"q": "What data type is x after: x = 3.14?", "a": "float", "difficulty": 0.2},
    ],
    1: [  # Conditions
        {"q": "Find the bug: if x = 5: print('yes')", "a": "Should be x == 5", "difficulty": 0.3},
        {"q": "What is the output of: print(not True)?", "a": "False", "difficulty": 0.3},
    ],
    2: [  # Loops
        {"q": "How many times does 'for i in range(5)' execute?", "a": "5 times (i=0..4)", "difficulty": 0.45},
        {"q": "What is wrong with: while i < 5: print(i)?", "a": "Missing i += 1 — infinite loop", "difficulty": 0.5},
    ],
    3: [  # Functions
        {"q": "What is the output of: def f(n): n=n+1; x=5; f(x); print(x)?", "a": "5", "difficulty": 0.55},
        {"q": "What does a function return if there is no return statement?", "a": "None", "difficulty": 0.5},
    ],
    4: [  # Recursion
        {"q": "What is wrong with: def countdown(n): print(n); countdown(n-1)?", "a": "Missing base case", "difficulty": 0.75},
        {"q": "What is the output of: def fact(n): return 1 if n==0 else n*fact(n-1); print(fact(4))?", "a": "24", "difficulty": 0.8},
    ],
    5: [  # Data Structures
        {"q": "What is the output of: a=[1,2,3]; b=a; b.append(4); print(a)?", "a": "[1,2,3,4]", "difficulty": 0.6},
        {"q": "What does stack.pop() return from [1,2,3]?", "a": "3", "difficulty": 0.6},
    ],
    6: [  # Algorithms
        {"q": "What is the time complexity of binary search?", "a": "O(log n)", "difficulty": 0.7},
        {"q": "What is the space complexity of merge sort?", "a": "O(n)", "difficulty": 0.75},
    ],
}

# Simulated wrong answers for demo (to trigger misconception extraction)
WRONG_ANSWERS = {
    0: ["x is still 5", "10", "error"],
    1: ["the code is correct", "yes and no both print", "True"],
    2: ["4 times", "1 to 5", "nothing is wrong it stops at 5"],
    3: ["6 because increment adds 1 to x", "it returns 0", "the function returns x"],
    4: ["nothing is wrong it counts to 0", "it will stop at 0", "returns 1"],
    5: ["[1,2,3] because b is a copy", "1", "nothing"],
    6: ["O(n)", "O(1) because it checks one element", "O(n^2)"],
}


# ─── Agent Implementations ────────────────────────────────────────────────────

class DiagnosisAgent(BaseAgent):
    """
    Conducts initial assessment and initializes the learner profile.
    Responsibility: gather preliminary evidence, set initial priors.
    """

    def __init__(self, **kw):
        super().__init__("DiagnosisAgent", **kw)
        self.rng = random.Random(42)

    def run(self, learner_id: str) -> LearnerState:
        state = LearnerState()
        # Initialize with low mastery priors, moderate uncertainty
        for i in range(N_CONCEPTS):
            state.mastery[i] = self.rng.uniform(0.15, 0.35)
            state.uncertainty[i] = 0.6
        return state


class AssessmentAgent(BaseAgent):
    """
    Selects assessment items for the target concept.
    Adapts difficulty based on current learner state.
    """

    def __init__(self, **kw):
        super().__init__("AssessmentAgent", **kw)
        self.rng = random.Random(123)

    def run(self, concept_id: int, learner_state: LearnerState, n_questions: int = 1) -> list[dict]:
        questions = QUESTION_BANK.get(concept_id, [{"q": "Describe this concept.", "a": "correct answer", "difficulty": 0.5}])
        selected = self.rng.sample(questions, min(n_questions, len(questions)))
        for q in selected:
            q["concept_id"] = concept_id
            q["concept_name"] = CONCEPT_NAMES[concept_id]
        return selected


class EvaluationAgent(BaseAgent):
    """
    Evaluates student answers and calls the LLM extraction component.
    Produces structured evidence (ExtractionResult) consumed by LearnerModelingAgent.
    """

    def __init__(self, llm_provider=None, **kw):
        super().__init__("EvaluationAgent", **kw)
        self.llm = llm_provider or MockLLMProvider()
        self.rng = random.Random(456)

    def run(
        self,
        question: dict,
        student_answer: str,
        learner_state: LearnerState,
    ) -> dict:
        concept_id = question.get("concept_id", 0)
        request = ExtractionRequest(
            question=question["q"],
            student_answer=student_answer,
            concept_name=question.get("concept_name", CONCEPT_NAMES[concept_id]),
            expected_answer=question.get("a"),
            concept_id=concept_id,
        )
        result = self.llm.extract(request)
        return {
            "correctness": result.correctness,
            "misconception_label": result.misconception_label,
            "misconception_severity": result.misconception_severity,
            "misconception_confidence": result.confidence,
            "reasoning_quality": result.reasoning_quality,
            "concept_id": concept_id,
            "difficulty": question.get("difficulty", 0.5),
        }


class LearnerModelingAgent(BaseAgent):
    """
    Runs UMiKT-GAT inference to update the learner state.

    This is the core agent — it calls the trained ML model and returns
    the updated four-part state S_i = [M_i, MC_i, U_i, R_i].

    The LLM extraction result is consumed here as input features,
    NOT used to directly set the learner state (preserving the LLM/ML separation).
    """

    def __init__(self, model, mc_wrapper, device="cpu", **kw):
        super().__init__("LearnerModelingAgent", **kw)
        self.model = model
        self.mc_wrapper = mc_wrapper
        self.device = device
        self._history: dict[str, list] = {}  # learner_id → interaction history

    def run(
        self,
        learner_id: str,
        eval_result: dict,
        learner_state: LearnerState,
    ) -> LearnerState:
        concept_id = eval_result.get("concept_id", 0)

        # Append to history
        if learner_id not in self._history:
            self._history[learner_id] = []
        self._history[learner_id].append(eval_result)

        # Build feature record
        rec = {
            "correctness": eval_result.get("correctness", 0.5),
            "difficulty": eval_result.get("difficulty", 0.5),
            "response_time_norm": 0.4,
            "attempts": 0.2,
            "temporal_gap_norm": 0.1,
            "evaluator_confidence": eval_result.get("misconception_confidence", 0.75),
            "question_type": "short_answer",
            "concept_id": concept_id,
            "misconception_severity": eval_result.get("misconception_severity", 0.0),
            "misconception_confidence": eval_result.get("misconception_confidence", 0.0),
            "reasoning_quality": eval_result.get("reasoning_quality", 0.5),
        }

        fv = build_feature_vector(rec)
        x_t = torch.tensor(fv, dtype=torch.float32, device=self.device)

        # Build sequence tensor (single interaction for now)
        from src.features.feature_builder import N_CONCEPTS, INPUT_DIM
        x_seq = torch.zeros(N_CONCEPTS, 10, INPUT_DIM, device=self.device)
        x_seq[concept_id, 0] = x_t

        ls_tensor = learner_state.to_tensor(self.device)

        # MC-Dropout inference for uncertainty
        def fwd():
            out = self.model(x_seq, ls_tensor)
            return out["mastery"]

        mean_m, var_m, _ = self.mc_wrapper.sample_predictions(fwd)

        # Update learner state
        new_state = LearnerState(
            mastery=learner_state.mastery.copy(),
            misconception_severity=learner_state.misconception_severity.copy(),
            misconception_class=learner_state.misconception_class.copy(),
            uncertainty=learner_state.uncertainty.copy(),
            retention=learner_state.retention.copy(),
            forgetting_risk=learner_state.forgetting_risk.copy(),
            interaction_count=learner_state.interaction_count.copy(),
        )

        for i in range(N_CONCEPTS):
            new_state.mastery[i] = float(mean_m[i].item())
            new_state.uncertainty[i] = min(1.0, float(var_m[i].item()) * 10)
            new_state.forgetting_risk[i] = 1.0 - new_state.retention[i]

        # Update misconception for the interacted concept
        mc_sev = eval_result.get("misconception_severity", 0.0)
        new_state.misconception_severity[concept_id] = 0.7 * learner_state.misconception_severity[concept_id] + 0.3 * mc_sev
        new_state.interaction_count[concept_id] += 1

        return new_state


class CurriculumPlanningAgent(BaseAgent):
    """Generates a learning path (ordered list of concepts) from the learner state."""

    def __init__(self, **kw):
        super().__init__("CurriculumPlanningAgent", **kw)
        self.priority_fn = CurriculumPriority()

    def run(self, learner_state: LearnerState, goal_relevance: list[float] | None = None) -> list[tuple[int, float]]:
        priorities = self.priority_fn.compute_priorities(
            mastery=learner_state.mastery,
            misconception_severity=learner_state.misconception_severity,
            uncertainty=learner_state.uncertainty,
            forgetting_risk=learner_state.forgetting_risk,
            goal_relevance=goal_relevance,
        )
        return self.priority_fn.top_concepts(priorities, n=3)


class CurriculumAdaptationAgent(BaseAgent):
    """Selects next curriculum action via the action policy."""

    def __init__(self, **kw):
        super().__init__("CurriculumAdaptationAgent", **kw)
        self.policy = ActionPolicy()
        self.priority_fn = CurriculumPriority()

    def run(self, learner_state: LearnerState, top_concepts: list[tuple[int, float]]) -> dict:
        if not top_concepts:
            return {"action": "practice_current_concept", "concept_id": 0, "explanation": "No priority concept identified."}

        concept_id, priority = top_concepts[0]
        dep_scores = self.priority_fn.compute_dependency_scores(learner_state.mastery)

        action, explanation = self.policy.select_action(
            concept_id=concept_id,
            mastery=learner_state.mastery[concept_id],
            misconception_severity=learner_state.misconception_severity[concept_id],
            uncertainty=learner_state.uncertainty[concept_id],
            forgetting_risk=learner_state.forgetting_risk[concept_id],
            dep_score=dep_scores[concept_id],
        )

        return {
            "action": action,
            "concept_id": concept_id,
            "concept_name": CONCEPT_NAMES[concept_id],
            "priority": round(priority, 4),
            "explanation": explanation,
        }


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class AgentOrchestrator:
    """
    Wires all agents and runs the closed adaptation loop.

    One closed loop = Assessment → Evaluation → Learner Modeling →
                      Graph Propagation (via GAT in model) → Curriculum Decision

    Logs all agent calls, LLM calls, and latencies for RQ6 system cost tracking.
    """

    def __init__(self, model, device="cpu", log_dir: Path | None = None):
        self.log_dir = log_dir or Path("experiments/results/agent_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        from src.models.mc_dropout_wrapper import MCDropoutWrapper
        mc_wrapper = MCDropoutWrapper(model, n_samples=10, device=str(device))
        llm = MockLLMProvider()

        self.diagnosis = DiagnosisAgent(log_dir=self.log_dir)
        self.assessment = AssessmentAgent(log_dir=self.log_dir)
        self.evaluation = EvaluationAgent(llm_provider=llm, log_dir=self.log_dir)
        self.learner_modeling = LearnerModelingAgent(model=model, mc_wrapper=mc_wrapper, device=device, log_dir=self.log_dir)
        self.curriculum_planning = CurriculumPlanningAgent(log_dir=self.log_dir)
        self.curriculum_adaptation = CurriculumAdaptationAgent(log_dir=self.log_dir)
        self.llm = llm

        self._session_log: list[dict] = []

    def run_session(
        self,
        learner_id: str,
        n_interactions: int = 5,
        simulate_wrong_answers: bool = True,
        rng_seed: int = 42,
    ) -> dict:
        """
        Run a complete adaptive learning session.

        Returns dict with per-step log and final learner state.
        """
        rng = random.Random(rng_seed)
        t_session_start = time.perf_counter()

        print(f"\n{'='*60}")
        print(f"UMiKT-GAT Demo Session — Learner: {learner_id}")
        print(f"NOTE: This session runs on SYNTHETIC/MOCK data.")
        print(f"{'='*60}\n")

        # Initial diagnosis
        learner_state = self.diagnosis(learner_id=learner_id)
        print(f"[Init] Learner state initialized.")

        step_log = []

        for step in range(1, n_interactions + 1):
            print(f"\n{'─'*40}")
            print(f"STEP {step}/{n_interactions}")

            # Plan: select target concept
            top = self.curriculum_planning(learner_state=learner_state)
            decision = self.curriculum_adaptation(learner_state=learner_state, top_concepts=top)
            concept_id = decision["concept_id"]

            print(f"  Target concept:  {decision['concept_name']}")
            print(f"  Priority:        {decision['priority']:.4f}")
            print(f"  Planned action:  {decision['action']}")

            # Assess
            questions = self.assessment(concept_id=concept_id, learner_state=learner_state)
            q = questions[0]

            # Simulate student answer
            if simulate_wrong_answers and rng.random() < 0.5:
                wrong_pool = WRONG_ANSWERS.get(concept_id, ["I don't know"])
                student_answer = rng.choice(wrong_pool)
            else:
                student_answer = q["a"]

            print(f"  Question:        {q['q'][:70]}...")
            print(f"  Student answer:  {student_answer[:60]}")

            # Evaluate
            eval_result = self.evaluation(
                question=q,
                student_answer=student_answer,
                learner_state=learner_state,
            )
            print(f"  Correctness:     {eval_result['correctness']:.3f}")
            print(f"  Misconception:   {eval_result['misconception_label']} "
                  f"(sev={eval_result['misconception_severity']:.3f})")

            # Update learner state
            learner_state = self.learner_modeling(
                learner_id=learner_id,
                eval_result=eval_result,
                learner_state=learner_state,
            )

            # Final action selection after state update
            top2 = self.curriculum_planning(learner_state=learner_state)
            final_decision = self.curriculum_adaptation(learner_state=learner_state, top_concepts=top2)

            print(f"\n  [After update]")
            print(f"  M[{decision['concept_name']}] = {learner_state.mastery[concept_id]:.3f}")
            print(f"  U[{decision['concept_name']}] = {learner_state.uncertainty[concept_id]:.3f}")
            print(f"  Action:    {final_decision['action']}")
            print(f"  Rationale: {final_decision['explanation']}")

            step_log.append({
                "step": step,
                "concept_id": concept_id,
                "concept_name": decision["concept_name"],
                "student_answer": student_answer,
                "correctness": eval_result["correctness"],
                "misconception_label": eval_result["misconception_label"],
                "misconception_severity": eval_result["misconception_severity"],
                "mastery_after": learner_state.mastery[concept_id],
                "uncertainty_after": learner_state.uncertainty[concept_id],
                "action": final_decision["action"],
                "explanation": final_decision["explanation"],
            })

        session_time = time.perf_counter() - t_session_start

        # Collect system cost
        all_agents = [
            self.diagnosis, self.assessment, self.evaluation,
            self.learner_modeling, self.curriculum_planning, self.curriculum_adaptation
        ]
        system_cost = {
            "session_total_latency_sec": round(session_time, 3),
            "n_interactions": n_interactions,
            "agents": [a.get_usage_stats() for a in all_agents],
            "llm_calls": self.llm.call_count,
            "llm_tokens": self.llm.token_count,  # 0 for mock provider
            "llm_provider": "MockLLMProvider (offline, no API calls)",
        }

        print(f"\n{'='*60}")
        print("SESSION COMPLETE")
        print(f"  Total latency:  {session_time:.2f}s")
        print(f"  LLM calls:      {system_cost['llm_calls']} (mock, 0 tokens)")
        print(f"{'='*60}")

        return {
            "learner_id": learner_id,
            "steps": step_log,
            "final_state": learner_state.summary(),
            "system_cost": system_cost,
        }
