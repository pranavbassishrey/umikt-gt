# Minor Project Comprehensive Research Report: Uncertainty-Aware Multidimensional Knowledge Tracing with Graph Attention Networks (UMiKT-GAT)

**Project Title:** UMiKT-GAT: Uncertainty-Aware Multidimensional Knowledge Tracing with Graph Attention Networks  
**Dataset:** ASSISTments / FoundationalASSIST (Real Human Student Interactions, Common Core Mathematics, Grades 6–8)  
**Primary Evaluation:** Strictly Aligned Next-Response Prediction on Disjoint Student Trajectories (11,700 Targets)  
**Execution Environment:** PyTorch 2.14.0+cu130 | NVIDIA GeForce RTX 3050 Laptop GPU | Linux x86_64  
**Repository Branch:** `main`  

---

## Executive Summary

This report documents the final experimental findings, architectural evolution, comparative benchmarks, and failure mode analyses for the **UMiKT-GAT** research project. 

The primary research inquiry investigated:
> *Can an educational knowledge-tracing architecture that integrates chronological sequence modeling, relational prerequisite graph propagation, and uncertainty estimation predict future learner correctness better than established baselines (BKT, DKT) on real-world student interaction data?*

### The Core Finding:
Under the reported learner-disjoint next-response evaluation protocol on `FoundationalASSIST`, **UMiKT-GAT v2 achieved a ROC-AUC of 0.8017**, compared with **0.7140 for DKT** and **0.6974 for BKT**. The reported results indicate improved discrimination and a lower Brier score for UMiKT-GAT v2. Both neural models were evaluated on the exact same 11,700 target interactions according to the repository's alignment audit. Further controlled ablations on real data are needed to establish the independent causal contribution of graph attention and quantify the effect of optimization improvements.

### Key Verified Results on Real Data:
1. **Predictive Performance:** UMiKT-GAT v2 achieved an **ROC-AUC of 0.8017** and **PR-AUC of 0.8508**, exceeding the standard **DKT baseline (ROC-AUC 0.7140, PR-AUC 0.7693)** by **+0.0877 AUC** and **BKT (ROC-AUC 0.6974)** by **+0.1043 AUC**.
2. **Learner-Clustered Statistical Significance:** To account for intra-student correlation across the 39 sequential predictions per learner, a **learner-level clustered bootstrap** ($B = 2,000$ resamples of 300 test learners with replacement) was conducted. The 95% clustered confidence interval on the AUC delta is strictly positive: **$[+0.0811, +0.0954]$ ($p < 0.0005$)**. A paired McNemar test on binary correctness confirmed statistical significance ($p = 9.27 \times 10^{-45}$).
3. **Probabilistic Accuracy & Calibration:** UMiKT-GAT v2 achieved a **Brier score of 0.1764**, a 15.6% reduction compared to DKT (**0.2090**) and 43.8% reduction compared to BKT (**0.3138**). Both neural models achieved excellent calibration (**ECE $\approx 0.0128$**); UMiKT-GAT v2's lower Brier score is driven by **higher resolution / sharpness** (assigning confident probabilities near 0 and 1 while maintaining calibration).
4. **Target Alignment Audit:** An exhaustive audit verified row-level concordance between DKT and UMiKT-GAT v2 across all 11,700 targets with zero identifier, concept, or label mismatches.
5. **Trade-offs & Resource Requirements:** UMiKT-GAT v2 introduces a ~6.6× parameter expansion over DKT (399,617 vs. 60,000 parameters), higher training compute (262.05s vs. 113.87s on GPU) compared to tabular BKT (2.76s on CPU), loss of BKT's closed-form psychometric parameters, and reliance on an externally authored prerequisite DAG.

---

## 1. Research Problem & Theoretical Motivation

Knowledge Tracing (KT) models the dynamic latent state of a learner’s understanding over time to forecast future performance and guide personalized curriculum sequencing.

```
       BKT (1995)                      DKT (2015)                     UMiKT-GAT v2 (Proposed)
  ┌──────────────────┐            ┌──────────────────┐            ┌──────────────────────────────┐
  │ Independent HMMs │            │   Recurrent NN   │            │ Chronological Shared GRU     │
  │ per Concept      │            │ (Black-box LSTM) │            │   + Prerequisite Graph GAT   │
  └────────┬─────────┘            └────────┬─────────┘            └──────────────┬───────────────┘
           │                               │                                     │
  • No cross-concept transfer    • Recurrent sequential memory  • Explicit relational inductive bias
  • Rigid 2-state latent Markov  • Unstructured concept tokens  • Dynamic graph attention context
  • Psychometric interpretability• Implicit co-occurrence only  • Calibrated probability estimates
```

### 1.1 Bayesian Knowledge Tracing (BKT)
BKT represents student mastery as a latent binary variable in a two-state Hidden Markov Model (HMM) parameterized by four values per skill:
- $P(L_0)$: Initial mastery prior
- $P(T)$: Probability of transitioning from unmastered to mastered
- $P(G)$: Guess probability (correct response despite non-mastery)
- $P(S)$: Slip probability (incorrect response despite mastery)

**Theoretical Constraints:**
- **Independence Assumption:** BKT treats every educational concept as an independent Markov chain. An interaction on *"Ratios and Proportional Relationships"* yields zero Bayesian update on *"Expressions and Equations"*, ignoring curriculum prerequisites.
- **Fixed Transition Dynamics:** Transition probability $P(T)$ is static and does not scale with learner interaction velocity, attempts, or temporal decay.

### 1.2 Deep Knowledge Tracing (DKT)
DKT replaced independent HMMs with high-capacity recurrent neural networks (LSTM), vectorizing input interactions as embedded $(c_t, y_t)$ pairs.

**Theoretical Comparison:**
- **Implicit vs. Explicit Concept Relationships:** DKT *can* learn concept co-occurrences indirectly from sequential data patterns. However, it treats concept IDs as arbitrary categorical indices without an explicit structural prior. On datasets with diverse concepts and sparse cross-skill transitions, the recurrent hidden state must divide capacity between temporal tracking and discovering relational structure from scratch.
- **Unstructured Projection:** DKT applies a linear projection from the hidden state $h_t$ across all concepts simultaneously, rather than routing context through an educational prerequisite graph.

### 1.3 The Proposed UMiKT-GAT Architecture
UMiKT-GAT integrates chronological sequence modeling with graph-based relational context:
1. **Shared Chronological Sequence Backbone:** A multi-layer GRU processes student interactions chronologically across all concepts to capture student learning speed, momentum, and response patterns.
2. **Relational Graph Attention Network (GAT):** An educational prerequisite DAG propagates concept mastery representations across related mathematical skills using multi-head attention.
3. **Target-Specific Next-Response Head:** To predict correctness on concept $c_{t+1}$, the network conditions on both the recurrent trajectory $h_t$ and the graph-propagated node representation of $c_{t+1}$.

---

## 2. Dataset & Experimental Setup

### 2.1 The FoundationalASSIST Dataset
All primary experiments were executed on real-world student interaction trajectories from the **FoundationalASSIST** benchmark (curated by ASSISTments from the middle school Illustrative Mathematics curriculum, Common Core Grades 6–8):

- **Dataset Scale:** 68,000 processed student interactions across middle-school mathematics.
- **Disjoint Student Cohorts:**
  - **Train:** 1,400 learners (56,000 interactions)
  - **Validation:** 300 learners (12,000 interactions)
  - **Test:** 300 learners (12,000 interactions)
  - **Student Split Overlap:** Exactly 0 (no student appears in more than one partition).
- **Sequence Properties:** Exactly 40 sequential interactions per learner across 7 core Common Core mathematics domains:
  1. Geometry
  2. Number and Operations — Fractions
  3. Expressions and Equations
  4. The Number System
  5. Ratios and Proportional Relationships
  6. Statistics and Probability
  7. Functions

```
                      7-Node Educational Prerequisite DAG
                      
         [Ratios & Proportional Relationships]
                          │
                          ▼
            [Expressions & Equations] ──────────► [Functions]
                          ▲                             ▲
                          │                             │
    [Number & Operations: Fractions] ──────────► [The Number System]
                          │
                          ▼
                     [Geometry]               [Statistics & Probability]
```

### 2.2 Leakage-Free Next-Response Evaluation Protocol
To guarantee experimental integrity:
- For each student sequence $t = 0, 1, \dots, 39$, the model receives observed interactions $[0, \dots, t-1]$ as historical input.
- The model outputs a predicted probability $\hat{y}_t \in (0, 1)$ for target concept $c_t$.
- Under no circumstances is the current response $y_t$ exposed to the model prior to prediction.
- Step 0 serves as the initial observed condition, yielding exactly **39 prediction targets per learner**.
- Across 300 test learners, the evaluation stream consists of exactly **$300 \times 39 = 11,700$ prediction targets**.

### 2.3 Row-Level Target Alignment Audit
A row-level audit verified that both neural models evaluated on identical data:
- **Composite Key Hash:** Unique hash `learner_id + "_" + str(interaction_id)`.
- **DKT vs. Ground Truth Label Matches:** **11,700 / 11,700 (100.0%, 0 mismatches)**.
- **UMiKT-v2 vs. Ground Truth Label Matches:** **11,700 / 11,700 (100.0%, 0 mismatches)**.
- **DKT vs. UMiKT-v2 Concept ID Matches:** **11,700 / 11,700 (100.0%, 0 mismatches)**.
- Verified artifacts: `scripts/verify_target_alignment.py`, `experiments/results/alignment_verification.json`.

---

## 3. Benchmark Results & Comparative Analysis

### 3.1 Primary Next-Response Benchmark on Real Data

All metrics below are computed on the 11,700 verified test predictions:

| Model | ROC-AUC | PR-AUC | Accuracy | RMSE | Brier Score | ECE (10 bins) | Parameters | Training Time | Test Set Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BKT Baseline** | 0.6974 | — | 0.6647 | 0.5602 | 0.3138 | — | **28** | **2.76s** (CPU) | <0.1s |
| **DKT Baseline (LSTM)** | 0.7140 | 0.7693 | 0.6713 | 0.4571 | 0.2090 | 0.0128 | ~60,000 | 113.87s (GPU) | 1.10s |
| **UMiKT-GAT v1 (Broken Baseline)** | 0.6114 | 0.6857 | 0.5899 | 0.6006 | 0.3607 | 0.3450 | 385,000 | 480.0s (GPU) | 15.20s |
| **UMiKT-GAT v2 (Proposed)** | **0.8017** | **0.8508** | **0.7313** | **0.4200** | **0.1764** | **0.0128** | 399,617 | 262.05s (GPU) | 12.54s |

*Hardware context: All neural models trained and evaluated on an NVIDIA GeForce RTX 3050 Laptop GPU (CUDA, PyTorch 2.14.0+cu130, batch size = 64). BKT fit on Intel CPU via Expectation-Maximization (25 iterations).*

```
                       ROC-AUC Performance Comparison
  1.00 ┌───────────────────────────────────────────────────────────────┐
       │                                                               │
  0.80 ┼───────────────────────────────────────────────────████ 0.8017 │
       │                                       ████ 0.7140 ░░░░        │
  0.70 ┼──────────████ 0.6974                 ░░░░         ░░░░        │
       │          ░░░░                        ░░░░         ░░░░        │
  0.60 ┼──────────░░░░────────────████ 0.6114─░░░░─────────░░░░────────┤
       │          ░░░░            ░░░░        ░░░░         ░░░░        │
  0.00 ┴──────────┴───┴───────────┴───┴───────┴───┴────────┴───┴────────┘
                   BKT         UMiKT-v1        DKT       UMiKT-v2 (Ours)
```

### 3.2 Statistical Significance: Interaction vs. Learner-Clustered Bootstrap

To evaluate whether the observed AUC advantage is statistically robust, we report both interaction-level and learner-clustered statistical analyses:

| Statistical Test | Unit of Resampling | Resamples ($B$) | Metric Delta ($\text{v2} - \text{DKT}$) | 95% Confidence Interval | $p$-value |
|---|---|:---:|:---:|:---:|:---:|
| **Standard Paired Bootstrap** | 11,700 individual interactions | 2,000 | +0.0877 AUC | $[+0.0797, +0.0956]$ | $p < 0.0005$ |
| **Learner-Clustered Bootstrap** | **300 test learners (39 targets each)** | **2,000** | **+0.0881 AUC** | **$[+0.0811, +0.0954]$** | **$p < 0.0005$** |
| **DKT Clustered Baseline CI** | 300 test learners | 2,000 | — | $[0.6971, 0.7288]$ | — |
| **UMiKT-v2 Clustered Baseline CI**| 300 test learners | 2,000 | — | $[0.7897, 0.8126]$ | — |
| **McNemar's Paired Test** | 11,700 paired binary outcomes | N/A | Discordant: 1598 vs 896 | $\chi^2 = 197.8$ | $p = 9.27 \times 10^{-45}$ |

**Interpretation:** Resampling by learner clusters accounts for potential correlation among predictions from the same student. The clustered 95% bootstrap CI on the AUC delta ($[+0.0811, +0.0954]$) remains strictly positive, and the 95% intervals for DKT ($[0.6971, 0.7288]$) and UMiKT-v2 ($[0.7897, 0.8126]$) do not overlap.

### 3.3 Calibration vs. Brier Score Decomposition

Brier score measures mean squared probability error:
$$\text{Brier} = \frac{1}{N}\sum_{i=1}^N (\hat{p}_i - y_i)^2$$
Under Murphy's decomposition, Brier score resolves into:
$$\text{Brier} = \text{Reliability (Calibration Error)} - \text{Resolution (Sharpness)} + \text{Uncertainty}$$

Both neural models achieved low Expected Calibration Error (**ECE $\approx 0.0128$** across 10 equal-width probability bins), demonstrating that both models produce well-calibrated probabilities. The 15.6% reduction in Brier score achieved by UMiKT-GAT v2 (0.1764 vs. 0.2090) is primarily driven by **higher resolution / sharpness**:

| Probability Bin | DKT Count | DKT Mean Pred | DKT True Rate | UMiKT-v2 Count | UMiKT-v2 Mean Pred | UMiKT-v2 True Rate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **$[0.0, 0.1)$** | 25 | 0.0666 | 0.1200 | **586** | 0.0496 | **0.0580** |
| **$[0.1, 0.2)$** | 294 | 0.1591 | 0.1803 | 591 | 0.1511 | 0.1591 |
| **$[0.2, 0.3)$** | 675 | 0.2554 | 0.2548 | 752 | 0.2523 | 0.2553 |
| **$[0.3, 0.4)$** | 935 | 0.3543 | 0.3572 | 935 | 0.3514 | 0.3572 |
| **$[0.4, 0.5)$** | 1,473 | 0.4548 | 0.4501 | 1,132 | 0.4509 | 0.4567 |
| **$[0.5, 0.6)$** | 1,840 | 0.5518 | 0.5386 | 1,218 | 0.5524 | 0.5361 |
| **$[0.6, 0.7)$** | 2,376 | 0.6508 | 0.6301 | 1,517 | 0.6518 | 0.6440 |
| **$[0.7, 0.8)$** | 2,317 | 0.7478 | 0.7376 | 1,854 | 0.7501 | 0.7174 |
| **$[0.8, 0.9)$** | 1,413 | 0.8409 | 0.8217 | 1,614 | 0.8448 | 0.8364 |
| **$[0.9, 1.0]$** | 352 | 0.9286 | 0.9062 | **1,501** | 0.9597 | **0.9467** |

**Observation:** DKT concentrates the majority of its predictions in the uncertain central bins ($[0.4, 0.8]$), assigning only 25 predictions below 0.1 and 352 above 0.9. UMiKT-GAT v2 makes significantly sharper probabilistic predictions (586 below 0.1 with 5.8% empirical correctness, 1,501 above 0.9 with 94.7% empirical correctness), maintaining low calibration error while separating mastered from unmastered states more decisively.

---

## 4. Empirical Performance Along the Student Trajectory

To evaluate performance across different sequence phases, we partitioned the 11,700 predictions by interaction timestep:

| Timestep Range | Prediction Count | DKT ROC-AUC | UMiKT-v2 ROC-AUC | AUC Delta | DKT Brier | UMiKT-v2 Brier |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Steps 1–2** | 600 | 0.6663 | **0.7983** | **+0.1320** | 0.2254 | **0.1795** |
| **Steps 3–5** | 900 | 0.7142 | **0.7891** | **+0.0749** | 0.2073 | **0.1836** |
| **Steps 6–10** | 1,500 | 0.7269 | **0.8007** | **+0.0738** | 0.1983 | **0.1721** |
| **Steps 11–20** | 3,000 | 0.6918 | **0.7753** | **+0.0836** | 0.2134 | **0.1859** |
| **Steps 21–39** | 5,700 | 0.7232 | **0.8161** | **+0.0929** | 0.2079 | **0.1711** |

### Empirical Insights:
1. **Early Steps (1–2):** UMiKT-GAT v2 maintains a +0.1320 AUC lead over DKT. This indicates that static concept difficulty priors and initial graph node embeddings provide useful inductive bias before long recurrent histories are established.
2. **Mid-Trajectory (3–20):** Both models improve as sequence history builds, with UMiKT-GAT v2 maintaining a consistent +0.07 to +0.08 AUC margin.
3. **Late-Trajectory (21–39):** UMiKT-GAT v2 peaks at **0.8161 AUC**, reflecting effective accumulation of cross-concept history.

---

## 5. Architectural Comparison: Proposed Mechanisms vs. Established Evidence

When evaluating why UMiKT-GAT v2 outperformed DKT and BKT, it is important to distinguish between **empirically observed differences** and **hypothesized causal mechanisms**:

```
      Chronological History                       Prerequisite Graph Prior
    [c_1, y_1] ──► [c_2, y_2] ──► [c_3, y_3]      c_parent ──(Prerequisite)──► c_target
                     │                                         │
                     ▼                                         ▼
            [Shared Sequence GRU]                  [Graph Attention Network (GAT)]
                     │                                         │
                     └───────────────────┬─────────────────────┘
                                         ▼
                            [Target Prediction Head]
                                         │
                                         ▼
                        p(y_target = 1) = 0.8017 AUC
```

### 5.1 Established Empirical Observations
1. **Observed Gain over BKT (+0.1043 AUC):** BKT's independent HMMs produce binary latent state estimates that saturate near 0.0 or 1.0, yielding high RMSE (0.5602) and Brier error (0.3138). Both neural models outperform BKT by modeling continuous sequential representations.
2. **Observed Gain over DKT (+0.0877 AUC):** UMiKT-GAT v2 demonstrates superior discrimination and probability resolution across all sequence phases on the same 11,700 targets.

### 5.2 Hypothesized Architectural Mechanisms (Requiring Real-Data Controlled Ablation)
- **Graph Attention Routing:** The intended mechanism is that GAT attention weights propagate mastery evidence along prerequisite edges (e.g. from *Fractions* to *Equations*). While theoretically motivated, the current experiments on real data evaluated the full v2 architecture as a package; isolating the exact numerical percentage of gain attributable specifically to GAT message passing (versus the shared GRU and optimization scheme) requires a real-data ablation removing only the GAT layer.
- **Relational vs. Black-Box Embeddings:** DKT projects all concepts simultaneously from a single hidden vector $h_t$, whereas UMiKT-GAT routes predictions through concept-specific node representations. This structural separation is the hypothesized driver of v2's higher resolution.

---

## 6. Technical Review & Failure Recovery: The v1 vs. v2 Story

A transparent account of the v1 failure and subsequent recovery is central to the project's engineering record:

### 6.1 The v1 Target Leakage Flaw
In early development (`src/run_foundational_assist.py`), UMiKT-GAT v1 reported an artificially high AUC of **0.9361**. Technical review revealed that the evaluation script sliced interactions by concept rather than globally, inadvertently placing the target interaction inside the sequence input window (`recs[-1]` included in `x_seq`). The model was essentially trained to copy the target response.

### 6.2 The v1 Architecture Collapse (AUC 0.6114)
When the leakage was eliminated and v1 was tested on strict next-response prediction, its performance collapsed to **0.6114 AUC** (worse than DKT's 0.7140). Two major design flaws were identified:
1. **Per-Concept GRU Channels:** v1 maintained 7 separate GRU channels (one per concept). Across a 40-step trajectory, each concept received only $\approx 5$ sparse interactions. The recurrent hidden state suffered from severe data sparsity and could not track chronological student momentum across concepts.
2. **Batching Stride Bug:** The training loop iterated with stride=20, exposing only ~1,000 examples per epoch out of 56,000 possible training interactions. The optimizer terminated at epoch 1 with minimal learning.

### 6.3 The v2 Architectural Recovery
UMiKT-GAT v2 resolved these flaws through:
- **Shared Sequence GRU (`SharedSeqGAT`):** A single chronological recurrent network encodes the student's unified learning trajectory across all concepts.
- **Dynamic Concept Context:** GAT concept embeddings are conditioned on cumulative concept exposure statistics.
- **Efficient Mini-Batching:** Vectorized batching (`pad_sequence` + `pack_padded_sequence`) expanded effective training throughput to 27,300 examples/epoch.
- **Cosine Annealing Scheduling:** Scheduled learning rate ($5 \times 10^{-4} \rightarrow 1 \times 10^{-5}$), achieving convergence at epoch 40 with a genuine, verified **0.8017 ROC-AUC**.

---

## 7. Where the Model Lacks: Limitations & Trade-offs

An honest appraisal indicates key areas where simpler baselines hold practical or theoretical advantages:

```
                            Trade-Off Radar
                            
                        Predictive AUC
                             1.00
                              /\
                             /  \
                            /    \
            Interpretability ────── Resource Efficiency
              (BKT Wins)             (DKT / BKT Win)
```

### 7.1 Computational Complexity & Parameter Footprint
- **Parameter Footprint:** UMiKT-GAT v2 has **399,617 parameters**, compared to ~60,000 in DKT and **28 parameters** in BKT (4 parameters $\times 7$ concepts).
- **Latency & Training Cost:**
  - BKT fits in **2.76 seconds** on CPU.
  - DKT trains in **113.87 seconds** and evaluates 11,700 predictions in 1.10 seconds.
  - UMiKT-GAT v2 requires **262.05 seconds** for training and **12.54 seconds** for evaluation on GPU (~1.07 ms per target interaction). GAT multi-head attention imposes $O(|V| \cdot d + |E| \cdot d)$ computation at every sequence step.
- **Multi-Agent Tutoring Overhead:** In multi-agent tutoring simulations (`system_cost.json`), the `LearnerModelingAgent` accounted for **99.5% of total system latency** (0.4436s out of 0.446s per 5-interaction block).

### 7.2 Loss of Closed-Form Psychometric Interpretability
- **BKT Advantage:** BKT parameters ($P(L_0), P(T), P(G), P(S)$) map directly to educational psychometrics. An educator can immediately inspect whether an error was due to a "slip" ($P(S)$) or genuine non-mastery.
- **Neural Model Limitation:** UMiKT-GAT states reside in an unconstrained continuous vector space $\mathbb{R}^{128}$. Translating these representations into qualitative diagnostic feedback requires auxiliary post-hoc probes.

### 7.3 Reliance on Prerequisite Graph Quality
- DKT discovers cross-skill patterns purely from empirical data.
- UMiKT-GAT's performance depends on the structural validity of the input prerequisite DAG (`foundational_assist_concept_graph.json`). If the pedagogical graph contains missing edges, incorrect dependencies, or cycles, GAT attention propagation may propagate biased representations across concepts.

### 7.4 Absence of Real Misconception Ground Truth
- While synthetic ablations explored multi-task misconception prediction, the real `FoundationalASSIST` dataset contains only binary correctness (`correctness`: 0 or 1), response times, and attempt counts.
- Evaluating whether graph-based knowledge tracing improves diagnostic misconception extraction on real student text responses remains unvalidated due to the absence of expert-annotated misconception labels in standard KT benchmarks.

---

## 8. Supplementary Synthetic Simulation Benchmark: Latent State Recovery

*Note: The results in this section evaluate continuous latent mastery recovery on synthetic data (`data/layer2_synthetic/`) and must NOT be confused with binary next-response prediction on real student data.*

In synthetic simulations, the ground truth includes an unobservable continuous mastery variable $m \in [0, 1]$ generated by a known generative process. Here, **Mastery RMSE** measures distance between continuous true mastery and predicted mastery ($\text{RMSE} = \sqrt{\mathbb{E}[(m - \hat{m})^2]}$), while **Brier score** is computed on separate question responses:

| Model Configuration | Synthetic Mastery AUC | Synthetic Mastery Accuracy | Synthetic Mastery RMSE | Synthetic Brier Score | Description |
|---|:---:|:---:|:---:|:---:|---|
| **Mastery-Only Baseline** | 0.6502 | 0.4740 | 0.1047 | — | Static prior baseline |
| **+ Temporal GRU** | 0.6309 | 0.4923 | 0.1052 | — | Temporal sequence without graph |
| **+ Graph (GAT Added)** | 0.6671 | 0.4641 | 0.1089 | — | Prerequisite message passing added |
| **Full UMiKT-GAT (Synthetic)** | 0.7387 | 0.5260 | 0.1089 | 0.3651 | Temporal + Graph + MC-Dropout |
| **Deterministic vs. MC-Dropout** | 0.7128 vs 0.7232 | 0.5007 vs 0.4909 | 0.1090 vs 0.1087 | — | 20-pass MC-Dropout yields +0.0104 AUC |

*Synthetic Ablation Insight:* In synthetic simulations with known prerequisite dependencies, adding graph attention improved latent state AUC from 0.6309 to 0.6671. However, validating the isolated contribution of the GAT layer on real-world student data remains the primary objective for follow-up work.

---

## 9. Conclusion & Research Outlook

### 9.1 Summary Conclusion
Under the reported learner-disjoint next-response evaluation protocol on `FoundationalASSIST`, UMiKT-GAT v2 achieved a ROC-AUC of 0.8017, compared with 0.7140 for DKT and 0.6974 for BKT. The reported results indicate improved discrimination and a lower Brier score for UMiKT-GAT v2. Both neural models were evaluated on the same 11,700 target interactions according to the repository's alignment audit. 

Learner-level clustered bootstrap analysis confirmed that the performance difference is statistically significant ($95\%\text{ CI: } [+0.0811, +0.0954]$, $p < 0.0005$). UMiKT-GAT v2 achieves this gain through higher probability resolution while maintaining low calibration error (ECE = 0.0128). These advantages involve clear engineering trade-offs: higher computational complexity (399K parameters, 262s training time), loss of closed-form psychometric interpretability, and dependence on an external prerequisite graph.

### 9.2 Prospective Research Directions
1. **Isolated Real-Data Graph Ablation:** Train an identical `SharedSeq` model on `FoundationalASSIST` with and without the GAT layer to isolate the exact causal contribution of prerequisite graph attention on real data.
2. **Differentiable Prerequisite Learning:** Learn the concept adjacency matrix jointly from learner transition patterns rather than relying on hand-authored static DAGs.
3. **Model Distillation for Edge Tutoring:** Distill the 399K-parameter UMiKT-GAT model into a compact recurrent student-state representation for low-latency (<5ms) client-side execution in interactive learning environments.

---

*Report compiled from authenticated experimental artifacts in `experiments/results/` and verified across 11,700 aligned test interactions on `FoundationalASSIST`.*
