# Sparse-Recovery Localization (COMP5340 contribution)

This module is the compressed-sensing contribution described in
`2026_06_18_compressed_sensing_autots_proposal_report_18.md`. It reformulates
the AutoTS **point-aggregation / localization-recovery** stage as **sparse
outlier recovery**

```
p_i = l_s + e_i + eps_i        (e_i = sparse gross outlier, eps_i = dense noise)
```

and compares the paper's NSAL against classic baselines and
compressed-sensing-inspired aggregators. It reuses the *validated* geometry from
`eval_table2_localization.py` (camera model, depth, WGS84 -> EPSG:3044
projection) so every method runs in the same coordinate space as the reproduced
Table II. As a check, `nsal` here reproduces the Table II AutoTS MAE (2.36 m).

## Methods (`aggregators.py`)

| Group | Methods | Core idea |
|---|---|---|
| Baselines | `mean`, `median`, `geometric_median`, `kmeans`, `dbscan` | classic robust / cluster centers |
| Paper | `nsal` | Gaussian affinity + MinCut + degree-weighted center |
| CS convex | `l1_sor` | `min ½‖p−l−e‖² + λ‖e‖₂` group-sparse, alternating min + group soft-threshold |
| CS greedy | `omp`, `cosamp`, `sp` | annihilator reduction `z = Fy`, block greedy pursuit, Random-Gaussian sensing |
| Proposed | `uspa` | NSAL degree weight × reliability `q_i` (depth/area/confidence) |

**k-guard (§4.10):** the sparse methods (`l1_sor/omp/cosamp/sp`) recover only
when `k ≥ K_MIN = 7`; below that they fall back to the geometric median and set
`fallback_triggered`. Greedy sparsity is capped at `⌊(k−1)/2⌋`.

Single entry point: `aggregators.aggregate(method, coords, meta, ...)` returns an
`AggResult(center, outlier_idx, outlier_scores, residual_norm, n_iter,
assumed_s, fallback_triggered)`.

## Experiments

Run individually:

```bash
cd AutoTS
python -m sparse_localization.experiments.exp1_main_table        # §9  main table
python -m sparse_localization.experiments.exp2_sparse_k          # §10 controlled k
python -m sparse_localization.experiments.exp3_outlier_stress    # §11 outlier stress
python -m sparse_localization.experiments.exp4_uspa_ablation     # §12 USPA cues
python -m sparse_localization.experiments.exp5_synthetic_phase   # §12C phase transition
```

Or all at once (also writes the reproducibility manifest `runs.csv`):

```bash
python -m sparse_localization.run_all          # all
python -m sparse_localization.run_all 1 4 5    # a subset
```

## Outputs (`AutoTS/results/cs/`)

| File | Level (§12B) | Content |
|---|---|---|
| `exp1/summary_exp1.csv` | aggregate | MAE/RMSE/R@1m/R@2m + fallback_rate per method |
| `exp1/per_sign_<method>.csv` | per-sign | est/gt/err, outlier support+scores, residual, fallback, runtime |
| `exp2/summary_exp2.csv`, `exp2/mae_table_exp2.csv` | aggregate | MAE mean±std by `(method, k)` |
| `exp3/summary_exp3.csv`, `exp3/mae_table_exp3.csv` | aggregate | MAE + outlier-detection P/R by `(magnitude, ratio, method)` |
| `exp4/summary_exp4.csv` | aggregate | metrics per reliability cue |
| `exp5/summary_exp5.csv`, `exp5/phase_<method>.csv` | aggregate | exact-recovery / support P-R / `l_err` by `(m, s)` |
| `runs.csv` | per-run | run id, timestamp, runtime, hyperparameters |

## Reading the results (honest interpretation)

- Full clean clouds (Exp 1): all robust methods land near 2.33–2.39 m MAE, with
  `cosamp` edging out the rest. Sparse recovery is not expected to beat NSAL
  here, because the recovered point clouds are already clean (proposal §14,
  Case 1).
- Sparse-k (Exp 2): at `k ∈ {2,3,5}` the sparse rows equal `geometric_median`
  exactly, which is the k-guard fallback firing by design (proposal `⟲`). MAE is
  not monotone across `k` rows, because each `k` filters a different sign subset.
  Compare within a row (across methods), not down a column.
- Outlier stress (Exp 3): `mean` degrades fastest with outlier ratio, while
  `geometric_median`, `median`, and `l1_sor` degrade slowest. `omp` is the
  weakest sparse method at high corruption (no backtracking).
- USPA cues (Exp 4): the cues move metrics only marginally, with depth helping
  R@2m and area helping R@1m. A useful, honest result, since the cues are not
  perfectly calibrated to localization error (proposal §14, Case 2).
- Phase transition (Exp 5): exact recovery sharpens as `m` grows and collapses
  when `2s+1 > m`. `cosamp` is strongest and `omp` weakest, matching CS theory
  (RIP and the error-correction bound).
