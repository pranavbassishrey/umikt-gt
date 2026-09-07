# UMiKT-GAT

**Uncertainty-Aware Misconception-Informed Knowledge Tracing using Graph Attention Networks**

A research codebase for the paper:
> *A Multi-Agent LLM Framework with an Uncertainty-Aware, Misconception-Informed Temporal Graph Knowledge Tracing Model for Autonomous Curriculum Adaptation*

> [!WARNING]
> **All quantitative results in this codebase are produced on SYNTHETIC data** unless
> the ASSISTments download in `data/layer1_public_kt/download_assistments.py` succeeds.
> Check `data/layer1_public_kt/layer1_metadata.json` — if `"is_real_data": false`, all
> Phase 3 and baseline metrics are synthetic fallback. See `DATASET_CARD.md`.

---

## Project Structure

```
umikt-gat/
├── data/
│   ├── concept_graph.json          # Prerequisite DAG (7 concepts)
│   ├── layer1_public_kt/           # ASSISTments download + fallback
│   ├── layer2_synthetic/           # Synthetic data generator + splits
│   └── layer2_handwritten/         # 50 hand-written Q&A pairs
├── src/
│   ├── data/loader.py              # Data loading and normalization
│   ├── features/feature_builder.py # 20-dim feature vector construction
│   ├── llm/                        # Schema, MockProvider, OpenAIProvider
│   ├── models/
│   │   ├── gru_encoder.py          # Temporal GRU encoder
│   │   ├── gat_layer.py            # Hand-rolled multi-head GAT
│   │   ├── fusion.py               # T→G and G→T fusion orders
│   │   ├── prediction_heads.py     # Mastery, Misconception, Retention heads
│   │   ├── mc_dropout_wrapper.py   # MC-Dropout uncertainty estimation
│   │   ├── retention_module.py     # Exponential decay + learned-λ
│   │   ├── umikt_gat.py            # Full model + ablation config factory
│   │   └── baselines/              # BKT and DKT baselines
│   ├── curriculum/priority.py      # Priority function + action policy
│   ├── agents/orchestrator.py      # All 7 agents + AgentOrchestrator
│   ├── train.py                    # Training script
│   ├── evaluate.py                 # ECE, Brier, reliability diagram
│   └── run_ablation.py             # Full ablation study master script
├── experiments/
│   ├── configs/                    # Per-variant YAML configs
│   └── results/                    # Generated metrics, CSVs, charts
├── tests/test_all.py               # 37 unit tests (all pass)
├── demo_session.py                 # Full closed-loop demo
├── DATASET_CARD.md
├── misconception_taxonomy.md
├── ARCHITECTURE.md
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic data
python data/layer2_synthetic/generate_synthetic.py

# 3. Attempt real data download (falls back to synthetic if unavailable)
python data/layer1_public_kt/download_assistments.py

# 4. Run unit tests (should be 37 passed)
python -m pytest tests/ -v

# 5. Run full ablation study (generates all results)
python src/run_ablation.py

# 6. Run the demo session (requires trained model from step 5)
python demo_session.py
```

---

## Model Overview

### Learner State

For each concept `i`, the model estimates:

```
S_i = [M_i, MC_i, U_i, R_i]
```

| Component | Description | Range |
|---|---|---|
| `M_i` | Knowledge mastery | [0,1] |
| `MC_i` | Misconception severity | [0,1] |
| `U_i` | Uncertainty (MC-Dropout variance) | [0,∞) |
| `R_i` | Retention (forgetting-adjusted) | [0,1] |

### Input Features

```
X_t = [correctness, difficulty, response_time_norm, attempts_norm,
        temporal_gap_norm, evaluator_confidence,
        question_type_one_hot (4d), concept_id_one_hot (7d),
        misconception_severity, misconception_confidence, reasoning_quality]
```

Total: **20-dimensional** feature vector.

### Architecture

```
X_t  →  GRU (shared across concepts)  →  H_t  [hidden_dim]
                                           ↓
                           Multi-Head GAT (Temporal→Graph)
                               Dynamic attention: α_ij,t = f(H_i, H_j, S_i, S_j, e_ij)
                                           ↓
                              Fusion: concat(H, Z) → MLP
                                           ↓
                    ┌──────────────────────┼──────────────────┐
                    ↓                      ↓                  ↓
               MasteryHead          MisconceptionHead   RetentionHead
               M_i ∈ [0,1]         7-class logits      R_i ∈ [0,1]
                    ↓
               MC-Dropout (20 passes) → U_i = Var[predictions]
```

### Multi-Task Loss

```
L_total = λ1·L_mastery + λ2·L_misconception + λ3·L_retention
```

- `L_mastery`: BCE (targets are continuous mastery in [0,1] binarized at 0.5)
- `L_misconception`: CrossEntropy with inverse-frequency class weights
- `L_retention`: MSE

### Curriculum Priority

```
P_i = α(1-M_i) + β·MC_i + γ·U_i + δ·Dep_i + ε·F_i + ζ·GoalRelevance_i
```

Weights α..ζ are **manually set heuristic priors** — not a novel contribution.

---

## Research Questions

| RQ | Question | Addressed by |
|---|---|---|
| RQ1 | Does the model improve mastery prediction vs baselines? | Ablation table, `ablation_results.csv` |
| RQ2 | Does misconception evidence improve learner-state estimation? | `+misconception` ablation variant |
| RQ3 | Does uncertainty-aware modeling improve curriculum decisions? | ECE/Brier in `eval_metrics.json` |
| RQ4 | Does prerequisite-aware graph modeling help? | `+graph` vs `+temporal` comparison |
| RQ5 | Does the curriculum mechanism improve learning? | Demo session (qualitative only) |
| RQ6 | What are the computational costs of LLM integration? | `system_cost.json` from demo session |

---

## Ablation Variants

| Variant | Temporal | Graph | Misconception | Uncertainty | Retention |
|---|---|---|---|---|---|
| Mastery-Only Baseline | ✗ | ✗ | ✗ | ✗ | ✗ |
| +Temporal | ✓ | ✗ | ✗ | ✗ | ✗ |
| +Graph | ✓ | ✓ | ✗ | ✗ | ✗ |
| +Misconception | ✓ | ✓ | ✓ | ✗ | ✗ |
| +Uncertainty | ✓ | ✓ | ✓ | ✓ | ✗ |
| Full UMiKT-GAT | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## Novelty Positioning

> **Do not claim individual components as novel.** GRU, GAT, MC-Dropout, and exponential
> forgetting models are all standard building blocks.

The proposed contribution under test is the **joint combination**:
1. Multi-dimensional learner state [M, MC, U, R] — four components jointly modeled
2. **Learner-state-conditioned prerequisite attention** (GAT-Dynamic): α_ij,t depends
   on both node hidden states and the current learner state, allowing prerequisite
   importance to vary per learner
3. Structured LLM extraction → ML integration (LLM provides evidence, not decisions)
4. Rule-based curriculum decision with multi-factor state (not just mastery)

Whether this combination produces measurable improvement over ablated baselines is
answered empirically by `run_ablation.py`.

---

## Limitations (Summary)

- All quantitative results are from **synthetic data** (see DATASET_CARD.md)
- Misconception taxonomy is **hand-authored**, domain-specific, not expert-validated
- LLM extraction uses **MockLLMProvider** (keyword-based) in all training/evaluation
- No real user study — RQ5 remains inconclusive without a prospective experiment
- MC-Dropout approximates, not equals, the true Bayesian posterior

See `DATASET_CARD.md` and `experiments/results/RESULTS_SUMMARY.md` for full details.
