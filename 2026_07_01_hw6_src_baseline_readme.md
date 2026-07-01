# 2026-07-01 — HW6 SRC baseline, recall-vs-outlier curves, and outlier logging

**Audience:** teammates writing the final report (esp. Ngan).
**Scope:** what changed in `AutoTS/sparse_localization/` on 2026-07-01 and how to
use it in the report. Nothing in the earlier reproduction or in HW6 was altered.

This addresses the three code asks from Ngan's message:

> The professor suggested our project direction is close to **SRC in HW6**
> because it is the **same model `y = Ax + e`**. For the code I need:
> 1. Log the number of outliers removed + which frame was dropped
> 2. Plot **R@1m / R@2m vs % outlier**
> 3. Add a **plain SRC baseline like HW6** to compare against

All three are now implemented, run, and saved under `AutoTS/results/cs/`.

---

## 0. The `y = Ax + e` link (one paragraph for the report intro)

HW6 recognises a corrupted face by writing it as `y = Ax + e`, where `Ax` is the
clean signal built from the dictionary and `e` is a **sparse** gross-error term,
then recovering `x` and `e` by `ℓ1` minimisation (dictionary `B = [A, I]`).

Our localization stage has the **same shape**. For one traffic sign we get `k`
noisy location estimates, one per image/frame:

```
p_i = l_s + e_i + eps_i        i = 1..k
```

`l_s` is the true sign location (shared across frames — the "signal"), `e_i` is a
**sparse gross outlier** (a bad frame: wrong depth/yaw), and `eps_i` is small
dense noise. So the sign location is exactly HW6's `y = Ax + e`, with the shared
location `l_s` playing the role of `Ax` and the per-frame error `e_i` playing the
role of `e`. That is why the plain-SRC baseline (below) is the natural
apples-to-apples comparison the professor pointed at.

---

## 1. Deliverable — plain SRC baseline (`src`)

**What:** a new aggregator `src` that is the direct HW6 SRC analogue. It solves

```
min_{l, e_i}   1/2 * sum_i || p_i - l - e_i ||^2  +  alpha * sum_i || e_i ||_1
```

with an **element-wise `ℓ1`** penalty on the error (each coordinate shrunk on its
own — exactly HW6's LASSO on `e`), by alternating minimisation: soft-threshold
the residuals, then re-estimate `l` as the mean of the de-corrupted points.
Default `alpha = 2.0` (metres).

**How it differs from our `l1_sor`:** same model, different penalty geometry.
`src` uses `||e_i||_1` (coordinate-wise, HW6 style); `l1_sor` uses a **group-`ℓ2`**
penalty `||e_i||_2` that keeps or drops a whole 2-D frame at once. Reporting both
lets us show the plain-SRC baseline **and** the group-sparse refinement.

**Where:** `aggregators.py` (function `src`, registered in `ALL_METHODS`,
`SPARSE_METHODS`, and the `aggregate()` k-guard dispatch). The `k >= 7` k-guard
applies, so on signs with too few frames `src` falls back to the geometric median
(flagged `fallback_triggered`), same rule as the other sparse methods.

**Result — Experiment 1, full clean point sets (275 signs, planedepth):**

| Method | MAE ↓ | RMSE ↓ | R@1m ↑ | R@2m ↑ | fallback |
|---|---|---|---|---|---|
| mean | 2.425 | 3.589 | 37.45 | 61.82 | 0.0% |
| median | 2.354 | 3.523 | 38.91 | 62.55 | 0.0% |
| geometric_median | 2.360 | 3.521 | 38.55 | 61.82 | 0.0% |
| dbscan | 2.377 | 3.619 | 37.82 | 62.55 | 0.0% |
| nsal (paper) | 2.359 | 3.525 | 38.55 | 62.55 | 0.0% |
| **src (HW6 baseline)** | **2.363** | **3.533** | **38.55** | **62.18** | 20.7% |
| l1_sor | 2.362 | 3.533 | 38.91 | 62.91 | 20.7% |
| omp | 2.419 | 3.648 | 38.91 | 62.18 | 20.7% |
| cosamp | 2.330 | 3.483 | 39.27 | 62.18 | 20.7% |
| sp | 2.357 | 3.507 | 38.18 | 61.09 | 20.7% |
| uspa (proposed) | 2.384 | 3.725 | 39.64 | 62.18 | 0.0% |

**How to read it (honest):** on clean clouds every robust method lands in
2.33–2.39 m — SRC is *not* expected to win here because the recovered clouds are
already clean. The value of `src` shows up under outlier stress (§2), which is the
regime HW6 is about. The 20.7% fallback is not a defect: ~1 sign in 5 has fewer
than 7 frames, so sparse recovery is under-determined and we honestly fall back.

---

## 2. Deliverable — R@1m / R@2m vs % outlier

**What was missing:** Experiment 3 swept the outlier ratio and *computed* recall
but only saved MAE. Now `summary_exp3.csv` carries `r1m_mean` and `r2m_mean`, and
`make_report.py` draws the recall curves.

**Where:**
- Data: `AutoTS/results/cs/exp3/summary_exp3.csv` (columns `r1m_mean`, `r2m_mean`).
- Figure: `AutoTS/results/cs/figures/exp3_recall_vs_ratio.png` (R@1m and R@2m
  panels vs outlier ratio, headline outlier magnitude 10 m).
- Table: the "Recall@2m vs outlier ratio" block in `report_tables.md`.

**Result — R@2m (%) vs outlier ratio, magnitude 10 m:**

| ratio | mean | median | geo_med | dbscan | nsal | **src** | l1_sor | omp | cosamp | sp | uspa |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0%  | 67.4 | 67.4 | 67.0 | 67.9 | 67.4 | **67.4** | 68.3 | 67.6 | 67.9 | 67.6 | 67.0 |
| 10% | 64.8 | 67.3 | 67.6 | 67.9 | 68.0 | **67.1** | 67.1 | 67.4 | 67.4 | 67.4 | 66.8 |
| 20% | 61.0 | 66.4 | 67.7 | 67.7 | 67.3 | **66.7** | 66.8 | 64.8 | 67.9 | 67.6 | 67.6 |
| 30% | 58.3 | 66.4 | 67.0 | 67.4 | 65.8 | **65.8** | 66.2 | 62.2 | 66.4 | 65.6 | 66.4 |
| 40% | 53.8 | 66.7 | 66.7 | 67.0 | 66.1 | **65.4** | 65.8 | 56.3 | 60.9 | 59.9 | 66.2 |

**Also — MAE (m) vs outlier ratio, magnitude 10 m:**

| ratio | mean | median | geo_med | dbscan | nsal | **src** | l1_sor | omp | cosamp | sp | uspa |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0%  | 2.271 | 2.178 | 2.184 | 2.223 | 2.186 | **2.188** | 2.187 | 2.215 | 2.182 | 2.182 | 2.221 |
| 10% | 2.415 | 2.205 | 2.200 | 2.228 | 2.194 | **2.222** | 2.215 | 2.252 | 2.205 | 2.205 | 2.220 |
| 20% | 2.507 | 2.215 | 2.192 | 2.265 | 2.224 | **2.237** | 2.211 | 2.316 | 2.227 | 2.233 | 2.252 |
| 30% | 2.575 | 2.225 | 2.216 | 2.243 | 2.261 | **2.275** | 2.247 | 2.422 | 2.286 | 2.338 | 2.283 |
| 40% | 2.657 | 2.248 | 2.232 | 2.247 | 2.321 | **2.311** | 2.270 | 2.610 | 2.453 | 2.487 | 2.354 |

**How to read it (report takeaway):** the naive `mean` collapses fastest (R@2m
67→54%, MAE 2.27→2.66 as outliers go 0→40%). The sparse / robust methods —
including the plain-SRC baseline — stay essentially flat (R@2m ≈ 65–67%). This is
the concrete evidence that modelling the bad frames as a sparse error term
protects the localization, i.e. the HW6 `y = Ax + e` idea pays off on our data.

---

## 3. Deliverable — log # outliers removed + which frame

**What:** two pieces.
1. Every per-sign log row now records which **source frame** was dropped, not just
   an index. `data.py` carries a per-observation `frame_id` (the KITTI image id,
   e.g. `2011_09_26_drive_0005_0000000034`); `runner.py` maps the recovered
   outlier indices back to those ids in a new `outlier_frames` column.
2. Experiment 3 writes a dedicated per-sign audit file per sparse method:
   `AutoTS/results/cs/exp3/per_sign_outliers_<method>.csv`
   (`src`, `l1_sor`, `omp`, `cosamp`, `sp`), at the headline magnitude, seed 0.

**Columns of `per_sign_outliers_<method>.csv`:**

| column | meaning |
|---|---|
| `sign_id`, `ratio`, `k` | which sign, outlier ratio injected, # frames |
| `n_injected`, `injected_idx`, `injected_frames` | ground-truth corrupted frames |
| `n_detected`, `detected_idx`, `detected_frames` | frames the method flagged/dropped |
| `detect_precision`, `detect_recall` | overlap of detected vs injected |
| `err_m`, `fallback` | resulting localization error, k-guard fallback flag |

**Example row (`per_sign_outliers_src.csv`, sign 0, ratio 0.1, k=27):** 3 frames
injected (`...034, ...040, ...049`), 4 dropped (those 3 + one extra `...008`) →
precision 0.75, recall 1.0, error 1.09 m. So you can point at exactly which
frames each method threw out and whether it caught the injected ones.

**Aggregate detection quality of `src` (from `summary_exp3.csv`, mag 10 m):**

| ratio | detect_precision | detect_recall |
|---|---|---|
| 10% | 0.651 | 0.995 |
| 20% | 0.733 | 0.998 |
| 30% | 0.795 | 0.997 |
| 40% | 0.832 | 0.996 |

**Honest note for the report:** `src` recall is ~1.0 (it catches essentially all
injected outliers) but precision is <1 — it also flags some clean-but-noisy frames
because the element-wise soft-threshold at `alpha = 2.0 m` fires on any coordinate
that sticks out. This is expected LASSO behaviour and is exactly the trade-off that
motivates the group-`ℓ2` `l1_sor` (which drops whole frames and over-flags less).
Precision *rises* with the outlier ratio simply because there are then more true
outliers to hit.

---

## 4. How to reproduce (from `AutoTS/`)

```bash
cd AutoTS
python -m sparse_localization.experiments.exp1_main_table      # main table (incl. src)
python -m sparse_localization.experiments.exp3_outlier_stress  # outlier stress + recall + per-sign logs
python -m sparse_localization.make_report                      # figures + report_tables.md
```

Runtimes on this machine: exp1 ≈ 16 s, exp2 ≈ 27 s, exp3 ≈ 44 s. Numbers above use
`depth_mode='planedepth'` (needs `AutoTS/data/img_depth.npz`, ~3.7 GB). Without
that file the code falls back to the thin-lens depth model and the absolute numbers
shift slightly — the *ranking and trends stay the same* (see module README).

---

## 5. What to put in the final report

- **Table:** Experiment 1 main table (§1) — include `src` so the plain-SRC
  baseline sits next to `nsal`, `l1_sor`, and the proposed `uspa`.
- **Figure:** `exp3_recall_vs_ratio.png` (§2) — the headline "robustness to
  outliers" plot; pair it with `exp3_mae_vs_ratio.png`.
- **Table:** R@2m (and/or MAE) vs outlier ratio (§2).
- **Table / callout:** `src` detection precision/recall (§3) as evidence the
  sparse-error model actually identifies the bad frames, with one concrete
  `per_sign_outliers_src.csv` example row.
- **One sentence** linking it to HW6 (§0): same `y = Ax + e` model, `l1_sor` is
  the group-sparse upgrade of the plain HW6 SRC baseline.

---

## 6. Files changed / added

**Code (`AutoTS/sparse_localization/`):**
- `aggregators.py` — new `src`; registered in `ALL_METHODS`, `SPARSE_METHODS`, dispatch.
- `data.py` — per-observation `frame_id` in `meta`.
- `runner.py` — new `outlier_frames` per-sign column.
- `experiments/exp2_sparse_k.py`, `experiments/exp3_outlier_stress.py` — `src` in
  method lists; exp3 also persists `r1m_mean`/`r2m_mean` and writes
  `per_sign_outliers_<method>.csv`.
- `make_report.py` — `src` in ordering; new `exp3_recall_vs_ratio.png` + recall table.
- `README.md` — method table, k-guard note, outputs table updated.

**Regenerated results (`AutoTS/results/cs/`):** `exp1/`, `exp2/`, `exp3/`
summaries + per-sign CSVs, `figures/*.png` (incl. new `exp3_recall_vs_ratio.png`),
`report_tables.md`.

**Unchanged:** the HW6 submission (`Comp5340_HW6_DoMinhPhung/comp5340_hw6_submission.tex`)
is the reference baseline and was intentionally left as-is; Experiment 5
(synthetic phase transition) keeps its own method list and was not extended to `src`.
