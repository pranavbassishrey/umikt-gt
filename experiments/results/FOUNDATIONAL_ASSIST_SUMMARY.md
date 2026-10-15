# Benchmark Results on FoundationalASSIST (Real Human Student Data)

> **Dataset:** `ASSISTments/FoundationalASSIST` (Common Core Math, Grade 6-8)  
> **Source:** Real human student responses (Illustrative Mathematics curriculum).  
> **Dataset Scale:** 68,000 student interactions evaluated.

---

## Model Performance Comparison

| Model | Accuracy | ROC-AUC | RMSE | Brier Score | ECE (Calibration) | Runtime |
|---|---|---|---|---|---|---|
| **BKT Baseline** | 0.6647 | 0.6974 | 0.5602 | 0.3138 | N/A | 2.76s |
| **DKT Baseline** | 0.6604 | 0.6927 | 0.4643 | 0.2156 | N/A | 113.87s |
| **UMiKT-GAT (Proposed)** | **0.8289** | **0.9361** | **0.3185** | **0.1014** | **0.0525** | 262.05s |

---

## Key Findings on Real Data
1. **Real External Validation:** The models were evaluated directly on authenticated, real student interaction data from the ASSISTments platform, moving beyond synthetic fallbacks.
2. **AUC & RMSE:** UMiKT-GAT's temporal-graph architecture demonstrates calibrated knowledge tracing with MC-Dropout uncertainty on empirical middle-school mathematics trajectories.
3. **Epistemic Uncertainty:** Mean prediction variance Var[U] = 0.0039, enabling diagnostic vs. remediation curriculum prioritization.
