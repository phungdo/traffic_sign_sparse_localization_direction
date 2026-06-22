# Table III — NSAL Ablation Study: Claims Verification

## Paper's Claims (Section V-D2)

The paper states three key findings from Table III:

> **1)** Our AutoTS outperforms both AutoTS w/o MinCut and AutoTS w/o Weight, indicating the effectiveness of these two components in our NSAL. The minimum cut helps eliminate noisy or inconsistent location points by preserving the most spatially coherent cluster. The weighted clustering further refines the estimated location by emphasizing high-confidence points, leading to more robust localization results.

> **2)** Our AutoTS outperforms both AutoTS-K-Means and AutoTS-DBSCAN, demonstrating the advantage of our NSAL in handling noisy and sparse location points compared to K-Means and DBSCAN.

> **3)** When dealing with sparse data, our AutoTS\* performs better than both AutoTS-K-Means\* and AutoTS-DBSCAN\*. This shows the improved sparse data processing ability of our NSAL.

---

## Full Results Comparison

### All Data (275 signs)

| Method | Paper MAE↓ | Our MAE↓ | Paper RMSE↓ | Our RMSE↓ | Paper R@1m↑ | Our R@1m↑ | Paper R@2m↑ | Our R@2m↑ |
|:-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| AutoTS w/o MinCut | 2.51 | 2.36 | 3.74 | 3.52 | 27.11 | 38.18 | 50.87 | 62.55 |
| AutoTS w/o Weight | 2.49 | 2.41 | 3.72 | 3.59 | 28.81 | 37.82 | 49.15 | 61.45 |
| AutoTS-K-Means | 2.56 | 2.43 | 3.82 | 3.59 | 27.11 | 37.45 | 47.46 | 61.82 |
| AutoTS-DBSCAN | 2.48 | 2.22 | 3.79 | 3.19 | 28.81 | 37.45 | 52.54 | 62.55 |
| **AutoTS (ours)** | **2.38** | **2.36** | **3.42** | **3.53** | **30.51** | **38.55** | **54.23** | **62.55** |

### Sparse Data (< 10 location points, 128 signs)

| Method | Paper MAE↓ | Our MAE↓ | Paper RMSE↓ | Our RMSE↓ | Paper R@1m↑ | Our R@1m↑ | Paper R@2m↑ | Our R@2m↑ |
|:-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| AutoTS-K-Means* | 2.46 | 2.31 | 3.52 | 3.24 | 28.81 | 35.94 | 52.54 | 58.59 |
| AutoTS-DBSCAN* | 2.42 | 2.28 | 3.48 | 3.20 | 30.51 | 35.94 | 52.54 | 59.38 |
| **AutoTS\*** | **2.36** | **2.28** | **3.37** | **3.18** | **32.20** | **35.94** | **54.23** | **59.38** |

---

## Claim-by-Claim Verification

### Claim 1: AutoTS > w/o MinCut and w/o Weight

**Paper's evidence (internally consistent ✅):**
- AutoTS (2.38) < w/o MinCut (2.51) → MinCut reduces MAE by 0.13m
- AutoTS (2.38) < w/o Weight (2.49) → Weighting reduces MAE by 0.11m

**Our reproduction ⚠️ Partially confirmed:**
- AutoTS (2.36) = w/o MinCut (2.36) → **MinCut effect is negligible** (0.00m difference)
- AutoTS (2.36) < w/o Weight (2.41) → **Weighting confirmed** (0.05m improvement)

**Verdict:** The weighted center component is confirmed as beneficial. However, MinCut provides no measurable improvement in our run, suggesting its benefit may be data- or seed-dependent.

---

### Claim 2: AutoTS > K-Means and DBSCAN

**Paper's evidence (internally consistent ✅):**
- AutoTS (2.38) < K-Means (2.56) → NSAL beats K-Means by 0.18m
- AutoTS (2.38) < DBSCAN (2.48) → NSAL beats DBSCAN by 0.10m

**Our reproduction ⚠️ Partially confirmed:**
- AutoTS (2.36) < K-Means (2.43) → **NSAL beats K-Means** ✅ (0.07m improvement)
- AutoTS (2.36) > DBSCAN (2.22) → **DBSCAN beats NSAL** ❌ (DBSCAN is 0.14m better!)

**Verdict:** NSAL's advantage over K-Means is confirmed. However, DBSCAN with optimized hyperparameters (eps=2.0, min_samples=1 via grid search) actually outperforms NSAL. The paper likely used fixed/sub-optimal DBSCAN parameters, which would explain why NSAL appeared superior in their experiments.

---

### Claim 3: Sparse data — AutoTS* > K-Means* and DBSCAN*

**Paper's evidence (internally consistent ✅):**
- AutoTS* (2.36) < K-Means* (2.46) → 0.10m improvement
- AutoTS* (2.36) < DBSCAN* (2.42) → 0.06m improvement

**Our reproduction ⚠️ Partially confirmed:**
- AutoTS* (2.28) < K-Means* (2.31) → **NSAL beats K-Means** ✅ (0.03m improvement)
- AutoTS* (2.28) = DBSCAN* (2.28) → **Tied** ⚠️

**Verdict:** NSAL maintains a slight advantage over K-Means on sparse data. However, its claimed advantage over DBSCAN vanishes when DBSCAN hyperparameters are properly tuned.

---

## Summary

| Claim | Paper Consistent? | Our Reproduction | Status |
|:------|:---:|:---|:---:|
| 1 — MinCut + Weight help | ✅ | Weight helps, MinCut negligible | ⚠️ Partial |
| 2 — NSAL > K-Means & DBSCAN | ✅ | NSAL > K-Means, but DBSCAN > NSAL | ⚠️ Partial |
| 3 — Sparse: NSAL* > others | ✅ | NSAL* > K-Means*, ties DBSCAN* | ⚠️ Partial |

### Key Insight

The paper's claims are internally consistent with their reported numbers. Our reproduction shows that **NSAL's advantage over DBSCAN is sensitive to DBSCAN hyperparameter tuning**. When DBSCAN is properly grid-searched, it can match or exceed NSAL. This doesn't invalidate the paper — the NSAL method has the advantage of being hyperparameter-free (no eps/min_samples to tune), which is a practical benefit the paper doesn't explicitly emphasize.

---

## Configuration

- DBSCAN params (grid-searched): eps=2.0, min_samples=1
- NSAL sigma: 2.5 (paper default)
- Sparse threshold: signs with < 10 location points (128/275 signs)
- Depth method: PlaneDepth (all variants)
- Evaluation: all samples (train + test, 275 signs with GT)
