# UMiKT-GAT Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         UMiKT-GAT System                                    │
│                                                                             │
│  ┌────────────────┐    ┌──────────────────┐    ┌──────────────────────────┐ │
│  │  LLM LAYER     │    │    ML LAYER       │    │   DECISION LAYER         │ │
│  │  (Evidence)    │───▶│  (State Model)    │───▶│  (Curriculum Policy)     │ │
│  │                │    │                  │    │                          │ │
│  │ MockLLMProvider│    │  UMiKTGATModel   │    │ CurriculumPriority       │ │
│  │ OpenAIProvider │    │  GRUEncoder       │    │ ActionPolicy             │ │
│  │                │    │  MultiHeadGAT    │    │ 6-factor priority fn     │ │
│  │ ExtractionResult    │  PredictionHeads │    │ 7-rule action selector   │ │
│  └────────────────┘    │  MCDropoutWrapper│    └──────────────────────────┘ │
│                        └──────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key invariant**: The LLM layer produces *evidence* (structured ExtractionResult). It does NOT
directly update the learner state. Only the ML layer updates S_i = [M_i, MC_i, U_i, R_i].
This separation is enforced in code: `EvaluationAgent.run()` → `ExtractionResult` → feature
injection into `LearnerModelingAgent.run()` → model forward pass.

---

## Feature Vector (20-dim)

```
Index  Field                    Source
0      correctness              LLM extraction (or raw binary)
1      difficulty               question metadata
2      response_time_norm       log(1+t_sec)/log(601), t∈[1,600]
3      attempts_norm            min(attempts/5, 1)
4      temporal_gap_norm        min(gap_hours/24, 1)
5      evaluator_confidence     LLM provider confidence
6-9    question_type_one_hot    {MC, short, code, debug}
10-16  concept_id_one_hot       concepts 0-6
17     misconception_severity   LLM extraction → MC severity
18     misconception_confidence LLM provider confidence (MC-specific)
19     reasoning_quality        LLM heuristic reasoning quality
```

---

## Model Architecture (full variant)

```
Input: X_t ∈ R^20 (per interaction)

┌─────────────────────────────────────────────────┐
│ GRU Encoder (shared weights across all concepts) │
│   input_dim = 20, hidden_dim = 64, layers = 2   │
│   MC-Dropout p=0.30 between layers              │
│   LayerNorm on output                            │
│   H_t ∈ R^64 per concept                        │
└──────────────────────────┬──────────────────────┘
                           │
                    Fusion Order A: T→G
                           │
┌──────────────────────────▼──────────────────────┐
│ Multi-Head GAT (4 heads, averaged)               │
│                                                  │
│ Static mode:                                     │
│   α_ij = softmax(LeakyReLU(W·[h_i‖h_j]))       │
│                                                  │
│ Dynamic mode (primary):                          │
│   α_ij,t = softmax(LeakyReLU(W·[h_i‖h_j‖     │
│                               s_i‖s_j‖e_ij]))  │
│ where s_i=[M_i,MC_i,U_i,R_i], e_ij=dep_weight │
│                                                  │
│ Heads averaged → ELU activation → residual       │
│ Z ∈ R^64 per concept                            │
└──────────────────────────┬──────────────────────┘
                           │
┌──────────────────────────▼──────────────────────┐
│ Fusion Layer                                     │
│   concat(H, Z) → Linear(128→64) → ReLU          │
│   → Dropout(0.30) → LayerNorm                   │
│   F ∈ R^64 per concept                          │
└──────────┬───────────────┬──────────────────────┘
           │               │
    ┌──────▼──────┐  ┌─────▼──────────┐  ┌────────▼────┐
    │ MasteryHead │  │ MisconceptionH. │  │ RetentionH. │
    │ Linear(64→32)  │ Linear(64→32)   │  │ Linear(64→32│
    │ →ReLU→Drop  │  │ →ReLU→Drop(0.2) │  │ →ReLU→Drop │
    │ →Linear(1)  │  │ →Linear(7)      │  │ →Linear(1) │
    │ →Sigmoid    │  │ (logits)        │  │ →Sigmoid   │
    │ M_i∈[0,1]  │  │ MC class∈{0..6} │  │ R_i∈[0,1] │
    └─────────────┘  └─────────────────┘  └────────────┘
           │
    MC-Dropout (20 forward passes, model.train() mode)
    U_i = Var[M̂_i^(1..20)] (predictive variance)
```

**Parameter counts:**
- Mastery-only baseline: 18,753
- +Temporal (GRU): 43,713
- +Graph (GAT): 57,573
- +Misconception: 59,884
- Full UMiKT-GAT: 61,997

---

## Concept Graph (Prerequisite DAG)

```
Variables (0)
    │
    ├──▶ Conditions (1) ──▶ Recursion (4) ◀──┐
    │                                         │
    └──▶ Loops (2) ──▶ Functions (3) ──▶ Data Structures (5)
                            │                         │
                            └─────────────────▶ Algorithms (6)
```

Edge features: `dependency_weight ∈ {0.7, 0.8, 0.9}` from `concept_graph.json`.
Used as scalar edge feature `e_ij` in the dynamic GAT attention computation.

---

## Learner State

```
S_i = [M_i, MC_i, U_i, R_i]

M_i  = mastery ∈ [0,1]                → from MasteryHead
MC_i = misconception severity ∈ [0,1] → from MisconceptionHead + LLM extraction
U_i  = uncertainty ∈ [0,∞)            → from MC-Dropout variance × 10 (normalized)
R_i  = retention ∈ [0,1]              → from RetentionHead
F_i  = 1 - R_i                        → forgetting risk (derived)
```

State update: full forward pass through UMiKTGATModel per interaction.
Initial state: uniform priors [M=0.3, MC=0, U=0.6, R=0.8].

---

## Curriculum Priority Function

```
P_i = α(1-M_i) + β·MC_i + γ·U_i + δ·Dep_i + ε·F_i + ζ·GoalRelevance_i

Default weights: α=0.35, β=0.25, γ=0.15, δ=0.10, ε=0.10, ζ=0.05
```

`Dep_i = mean(1 - M_prereq)` over prerequisites of concept i.

**These weights are manually set heuristic priors, not learned.** Learning them
(e.g., contextual bandit, multi-objective optimization) is left for future work.

---

## Action Policy (7 rules, ordered)

```
IF M_i ≥ 0.75 AND U_i < 0.40 AND MC_i < 0.50 AND F_i < 0.40
    → advance_to_next_topic

ELIF M_i ≥ 0.75 AND F_i ≥ 0.40
    → schedule_spaced_revision

ELIF M_i < 0.55 AND MC_i ≥ 0.50
    → misconception_targeted_content

ELIF M_i < 0.40 AND Dep_i ≥ 0.50
    → diagnose_prerequisite

ELIF U_i ≥ 0.40
    → adaptive_diagnostic_assessment

ELIF M_i < 0.40
    → misconception_targeted_remediation

ELSE
    → practice_current_concept
```

All thresholds are heuristic and documented as such. The policy is fully interpretable:
each action includes a human-readable rationale string computed from the current state.

---

## Multi-Agent System

```
┌──────────────────────────────────────────────────┐
│                  AgentOrchestrator               │
│                                                  │
│  ┌───────────┐    ┌──────────────┐              │
│  │ Diagnosis  │    │  Assessment  │              │
│  │ Agent     │    │  Agent       │              │
│  └─────┬─────┘    └──────┬───────┘              │
│        │ LearnerState     │ Question             │
│        ▼                  ▼                      │
│  ┌──────────────────────────────────────────┐    │
│  │           EvaluationAgent                │    │
│  │   answer → MockLLMProvider.extract()     │    │
│  │   → ExtractionResult (validated schema)  │    │
│  └──────────────────┬───────────────────────┘    │
│                     │ ExtractionResult            │
│                     ▼                            │
│  ┌──────────────────────────────────────────┐    │
│  │        LearnerModelingAgent              │    │
│  │   UMiKTGATModel.forward(x_seq, S)       │    │
│  │   MCDropoutWrapper (20 passes)           │    │
│  │   → Updated LearnerState S'             │    │
│  └──────────────────┬───────────────────────┘    │
│                     │ Updated S                  │
│                     ▼                            │
│  ┌────────────────────────────────────────────┐  │
│  │  CurriculumPlanningAgent                   │  │
│  │  → priority list (top 3 concepts)          │  │
│  └──────────────────┬─────────────────────────┘  │
│                     │                            │
│                     ▼                            │
│  ┌────────────────────────────────────────────┐  │
│  │  CurriculumAdaptationAgent                 │  │
│  │  → (action, explanation, concept_id)       │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

**Every agent call is logged** with timestamp, latency, and cumulative cost.
System cost for a 5-step session: 0.49s total, 5 LLM calls (mock, 0 API tokens).

---

## Uncertainty Estimation

**Method**: Monte Carlo Dropout (Gal & Ghahramani 2016)

```python
model.train()  # enable dropout
predictions = [model(x)["mastery"] for _ in range(20)]
mean = stack(predictions).mean(0)
variance = stack(predictions).var(0)  # epistemic uncertainty proxy
```

**Justification over alternatives:**
- Deep Ensembles: require N× training time; too expensive for this scale
- Evidential DL: requires modifying loss function; higher complexity
- Variational BNNs: requires reparameterization trick; significant refactoring
- MC-Dropout: reuses existing dropout, same training procedure, 20× inference cost

**Known limitation**: MC-Dropout approximates the variational posterior, not the true
Bayesian posterior. It tends to underestimate uncertainty in high-confidence regions.
The variance correlates with, but does not equal, true epistemic uncertainty.

---

## Training

**Optimizer**: Adam, lr=1e-3, weight_decay=1e-4
**Scheduler**: ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5)
**Gradient clipping**: max_norm=1.0
**Early stopping**: patience=10 epochs on val mastery RMSE
**Multi-task loss**:
  ```
  L = 1.0·BCE(M_hat, M_gt) + 0.5·CE(MC_hat, MC_gt, weights) + 0.3·MSE(R_hat, R_gt)
  ```
  - Class weights for misconception head: inverse-frequency from training distribution
  - Weights λ1=1.0, λ2=0.5, λ3=0.3 are heuristic starting points

---

## Verified Test Results (Fast Mode, 5 Epochs, CUDA)

> All on SYNTHETIC data. See DATASET_CARD.md.

| Metric | Value | Notes |
|---|---|---|
| Mastery Accuracy | 1.000 | Binarized at 0.5; synthetic data has clean separation |
| Mastery RMSE | 0.045 | Regression on continuous mastery in [0,1] |
| ECE | 0.021 | Well-calibrated (lower is better; 0 = perfect) |
| Brier Score | 0.002 | Very low; reflects synthetic data regularity |
| MC F1 (macro) | 0.194 | Low — class 0 (none) dominates; 5 epochs insufficient |
| MC Accuracy | 0.937 | Dominated by class 0 majority |
| Retention RMSE | 0.039 | Good fit; synthetic targets are smooth |
| DKT AUC | 0.978 | Strong baseline on clean synthetic data |
| BKT AUC | 0.970 | Competitive — synthetic data is HMM-like (BKT-friendly) |

> [!NOTE]
> Misconception F1 is low (0.194) because 5 epochs is insufficient for the minority
> class heads to converge. The full 50-epoch run (`python src/run_ablation.py`) will
> produce more meaningful misconception metrics.
