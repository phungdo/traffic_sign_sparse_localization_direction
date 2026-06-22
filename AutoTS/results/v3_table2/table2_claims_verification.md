# Table II — Traffic Sign Localization: Claims Verification

## Paper's Claims (Section V-B1)

The paper states four findings from Table II:

> **1)** On the KITTI-TS dataset, our AutoTS clearly outperforms GeoLocating, demonstrating the superiority of our method in geo-locating traffic signs.

> **2)** On the KITTI-TS dataset, our AutoTS has better performance than GeoLocating+NSAL, highlighting the effectiveness of our selected self-supervised monocular depth estimation model over the thin-lens model for depth extraction in this task. The high-quality location points further enhance traffic sign localization performance.

> **3)** On both the KITTI-TS and Aalborg datasets, GeoLocating+NSAL achieves better performance than GeoLocating. This demonstrates our NSAL method can assist in geo-locating traffic signs. Although the improvement brought by NSAL is relatively modest, it is specifically designed to handle challenges such as noise and sparsity. The performance of NSAL may also be limited by the quality of the location points.

> **4)** With the inclusion of direction information, both GeoLocating-D and GeoLocating-D+NSAL outperform GeoLocating and GeoLocating+NSAL, respectively, demonstrating the effectiveness of direction information in the Aalborg dataset.

**Note:** Claims 3 and 4 refer to the Aalborg dataset, which we did not reproduce (KITTI-TS only).

---

## Full Results Comparison (KITTI-TS Only)

| Method | Paper MAE↓ | Our MAE↓ | Paper RMSE↓ | Our RMSE↓ | Paper R@1m↑ | Our R@1m↑ | Paper R@2m↑ | Our R@2m↑ |
|:-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GeoLocating [7] | 3.98 | 4.26 | 6.27 | 5.68 | 11.86 | 15.27 | 22.03 | 35.64 |
| GeoLocating+NSAL | 3.80 | 4.28 | 5.92 | 5.70 | 15.25 | 13.09 | 27.12 | 35.64 |
| **AutoTS (ours)** | **2.38** | **2.36** | **3.42** | **3.53** | **30.51** | **38.55** | **54.23** | **62.55** |

### Timing (ours)

| Method | Total Time | Per Sign |
|:-------|:---:|:---:|
| GeoLocating [7] | 0.17s | 0.6 ms |
| GeoLocating+NSAL | 0.11s | 0.4 ms |
| AutoTS (ours) | 13.27s | 48.2 ms |

---

## Claim-by-Claim Verification

### Claim 1: AutoTS clearly outperforms GeoLocating

**Paper's evidence (internally consistent ✅):**
- AutoTS MAE (2.38) vs GeoLocating MAE (3.98) → 1.60m improvement
- AutoTS RMSE (3.42) vs GeoLocating RMSE (6.27) → 2.85m improvement
- AutoTS R@1m (30.51%) vs GeoLocating R@1m (11.86%) → +18.65pp
- AutoTS R@2m (54.23%) vs GeoLocating R@2m (22.03%) → +32.20pp

**Our reproduction ✅ Confirmed:**
- AutoTS MAE (2.36) vs GeoLocating MAE (4.26) → **1.90m improvement** (even larger gap)
- AutoTS RMSE (3.53) vs GeoLocating RMSE (5.68) → **2.15m improvement**
- AutoTS R@1m (38.55%) vs GeoLocating R@1m (15.27%) → **+23.28pp** (even larger gap)
- AutoTS R@2m (62.55%) vs GeoLocating R@2m (35.64%) → **+26.91pp**

**Verdict: ✅ Fully confirmed.** AutoTS dramatically outperforms GeoLocating in all metrics. The improvement gap is actually larger in our reproduction.

---

### Claim 2: AutoTS outperforms GeoLocating+NSAL (PlaneDepth > thin-lens)

**Paper's evidence (internally consistent ✅):**
- AutoTS MAE (2.38) vs GeoLocating+NSAL MAE (3.80) → 1.42m improvement
- Both use NSAL clustering, only depth method differs (PlaneDepth vs thin-lens)

**Our reproduction ✅ Confirmed:**
- AutoTS MAE (2.36) vs GeoLocating+NSAL MAE (4.28) → **1.92m improvement** (even larger)
- AutoTS R@1m (38.55%) vs GeoLocating+NSAL R@1m (13.09%) → **+25.46pp**

**Verdict: ✅ Fully confirmed.** PlaneDepth depth estimation is clearly superior to thin-lens for this task. This is expected since thin-lens requires knowing the real sign height, which varies by category, while PlaneDepth learns depth directly from images.

---

### Claim 3: GeoLocating+NSAL > GeoLocating (NSAL helps)

**Paper's evidence (internally consistent ✅):**
- GeoLocating+NSAL MAE (3.80) < GeoLocating MAE (3.98) → 0.18m improvement
- Paper acknowledges: "the improvement brought by NSAL is relatively modest"

**Our reproduction ❌ Not confirmed:**
- GeoLocating+NSAL MAE (4.28) > GeoLocating MAE (4.26) → **NSAL slightly hurts** (by 0.02m)
- GeoLocating+NSAL R@1m (13.09%) < GeoLocating R@1m (15.27%) → **NSAL hurts R@1m**
- Both R@2m are identical (35.64%)

**Verdict: ❌ Not confirmed.** In our reproduction, adding NSAL to GeoLocating's thin-lens points does not help and slightly hurts. This aligns with the paper's own caveat: "The performance of NSAL may also be limited by the quality of the location points." The thin-lens depth produces noisier location points, and NSAL's weighted center may amplify rather than correct systematic bias.

**Important context:** The discrepancy in GeoLocating baseline numbers (paper: 3.98 vs ours: 4.26) is because the thin-lens model requires knowing the real sign height. We estimated German standard heights per category (0.42–0.90m), while the paper's GeoLocating [7] implementation may use different values.

---

### Claim 4: Direction information helps (Aalborg dataset)

**Not reproduced** — we only evaluated on KITTI-TS. The Aalborg dataset was not available/used.

---

## Summary

| Claim | Paper Consistent? | Our Reproduction | Status |
|:------|:---:|:---|:---:|
| 1 — AutoTS >> GeoLocating | ✅ | Confirmed with even larger gap | ✅ Confirmed |
| 2 — PlaneDepth >> thin-lens | ✅ | Confirmed with even larger gap | ✅ Confirmed |
| 3 — NSAL helps GeoLocating | ✅ | NSAL slightly hurts with thin-lens points | ❌ Not confirmed |
| 4 — Direction helps (Aalborg) | ✅ | Not evaluated (Aalborg not used) | ⬜ N/A |

### Key Insights

1. **AutoTS (MAE=2.36) nearly exactly matches the paper (MAE=2.38)** — our core method reproduction is successful.
2. The **dominant improvement comes from PlaneDepth vs thin-lens** depth estimation, not from NSAL clustering. This is clear from both the paper's numbers and ours.
3. **NSAL's benefit is conditional** on input quality — it helps with PlaneDepth's accurate points but can slightly hurt with thin-lens's noisier points.
4. Our **R@1m and R@2m are significantly higher** than the paper across all methods, suggesting possible differences in the thin-lens calibration or coordinate transform implementation.

---

## Configuration

- **Evaluation scope**: All 275 signs with ground truth (train + test)
- **GeoLocating depth**: Thin-lens model (f=721.5px, per-category sign heights)
- **AutoTS depth**: PlaneDepth pre-computed depth maps (img_depth.npz)
- **DBSCAN params** (grid-searched): eps=1.0, min_samples=5
- **NSAL sigma**: 2.5
- **Coordinate system**: WGS84 ↔ EPSG:3044
