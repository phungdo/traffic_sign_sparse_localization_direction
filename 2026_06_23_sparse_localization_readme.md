# Today's Work: Sparse-Recovery Localization (2026-06-23)

**Team handoff. Read the TL;DR, then jump to your section.** This covers *only*
what we added today: the compressed-sensing contribution our proposal promised.
The AutoTS *reproduction* (detector, localization, orientation, figures, map)
was already done; see `reproduce_readme.md` for that.

---

## TL;DR (30 seconds)

1. We built the CS contribution the proposal describes: a clean module that
   takes each sign's noisy location points and recovers the true location with
   L1-SOR, OMP, CoSaMP, SP, and USPA, alongside the paper's NSAL and the classic
   baselines, plus a k-guard safety rule.
2. We ran all 5 experiments from the proposal. They work end-to-end, and the
   results behave the way compressed-sensing theory predicts.
3. We auto-generated the filled-in tables and 6 figures, ready to paste into the
   report.

Everything lives in `AutoTS/sparse_localization/`. Results in `AutoTS/results/cs/`.

One sanity check proves it is wired correctly: our `nsal` here gives MAE 2.36 m,
identical to the reproduced Table II AutoTS. Same data, same coordinate space,
so every new method is directly comparable to the paper.

---

## The big picture (1 picture)

```
AutoTS pipeline:  image+GPS -> detect -> depth -> location points -> [AGGREGATE] -> final location
                                                                      ^^^^^^^^^^^
                                                          THIS is our whole job.
```

Each sign gives several noisy points. We model each point as

```
point_i = true_location + sparse_gross_error_i + small_noise_i
```

and the task is to recover `true_location`. That "sparse gross error" framing is
the bridge to COMP5340 (compressed sensing). The paper solved it with graph
density (NSAL); we reframe it as sparse outlier recovery and test CS methods.

---

## Who looks at what (4-person split)

| If you're presenting | Read this section | Key file |
|---|---|---|
| The methods and math | "The 11 methods" + proposal §3–5 | `aggregators.py` |
| The experiments and results | "The 5 experiments" | `results/cs/report_tables.md` |
| The figures and slides | "Figures" | `results/cs/figures/*.png` |
| The reproducibility and logging | "How to run" + "What we log" | `runner.py`, `runs.csv` |

Nobody needs to read all the code. The README inside the module
(`AutoTS/sparse_localization/README.md`) is the deeper reference if you want it.

---

## The 11 methods (plain-English glossary)

One sentence each; the math is in the proposal section noted above.

| Method | One-line idea | Type |
|---|---|---|
| `mean` | average of all points | baseline |
| `median` | coordinate-wise middle | baseline (robust) |
| `geometric_median` | the 2D point minimizing total distance | baseline (robust) |
| `kmeans` | cluster, keep the biggest cluster's center | baseline |
| `dbscan` | density cluster, drop noise points | baseline |
| `nsal` | paper method: similar points vote, outliers cut, weighted center | paper |
| `l1_sor` | assume few points are badly wrong; solve for them with an L1 penalty | CS convex |
| `omp` / `cosamp` / `sp` | greedily *find which points are the outliers*, average the rest | CS greedy |
| `uspa` | NSAL, but trust points from big/near/confident detections more | our extension |

The one rule everyone should know is the k-guard. Sparse recovery only works if
there are enough clean points to outvote the bad ones. So if a sign has fewer
than 7 points, we do not pretend: we fall back to the geometric median and log
it. This happens for about 20% of signs. It is intentional, and it is the honest
CS story, since recovery guarantees need enough measurements.

---

## The 5 experiments (what each one proves)

Full numbers in `AutoTS/results/cs/report_tables.md`. The headlines:

**Exp 1, main table (all methods, full data).** All robust methods land at
2.33–2.39 m MAE, with `cosamp` marginally best (2.330). Sparse recovery does not
beat NSAL here, and that is expected: with all points kept, the clouds are
already clean. For the talk, the line is that everything ties on clean dense
data, so our methods earn their keep on the hard cases instead.

**Exp 2, few observations (k = 2,3,5,10,20,all).** At k = 2,3,5 the sparse
methods exactly equal the geometric median, which is the k-guard firing. State
one caveat up front: MAE is not comparable across k rows, because each k keeps a
different subset of signs. Compare methods within a row.

**Exp 3, injected outliers (0–40% of points corrupted).** This is the experiment
that shows robustness. `mean` degrades fastest (2.27 to 2.66), while `median`,
`geometric_median`, and `l1_sor` barely move (about 2.23–2.27 at 40%). `omp` is
the weakest sparse method under heavy corruption.

**Exp 4, USPA reliability cues (depth, area, confidence).** The cues move the
numbers only slightly: depth helps R@2m, area helps R@1m. Present this as a
useful negative result rather than a failure. The cues are not perfectly
calibrated to localization error, and saying so plainly is the honest read.

**Exp 5, synthetic phase transition (ground-truth outliers known).** Recovery is
perfect when there are enough observations and collapses when
`2·outliers + 1 > observations`, with CoSaMP strongest and OMP weakest. This is
the cleanest evidence that the methods match CS theory (RIP and the
error-correction bound), so it deserves its own slide.

---

## Figures (ready for slides)

In `AutoTS/results/cs/figures/`:

| File | Use it for |
|---|---|
| `exp1_main_table.png` | "all methods, full data" bar chart |
| `exp2_mae_vs_k.png` | error vs number of observations |
| `exp3_mae_vs_ratio.png` | robustness curves for Exp 3 |
| `exp3_heatmaps.png` | MAE heatmaps per outlier magnitude |
| `exp4_uspa_ablation.png` | reliability-cue comparison |
| `exp5_phase_transition.png` | the phase-transition heatmaps |

---

## How to run (copy-paste, from `AutoTS/`)

```bash
# everything (methods + 5 experiments), then build tables + figures
python -m sparse_localization.run_all
python -m sparse_localization.make_report
```

```bash
# or one experiment at a time
python -m sparse_localization.experiments.exp1_main_table
python -m sparse_localization.experiments.exp2_sparse_k
python -m sparse_localization.experiments.exp3_outlier_stress
python -m sparse_localization.experiments.exp4_uspa_ablation
python -m sparse_localization.experiments.exp5_synthetic_phase
```

Runtimes: exp2 about 25 s, exp3 about 38 s, the rest are seconds. No GPU, no
training.

---

## What we log (for the reproducibility section)

Three levels, as the proposal §12B asks:

- `results/cs/runs.csv`: one row per run, with timestamp, runtime, and
  hyperparameters.
- `results/cs/exp1/per_sign_<method>.csv`: one row per sign per method, with the
  predicted vs GT location, the error, which observations were flagged as
  outliers, the residual, the iteration count, whether the k-guard fired, and
  the runtime.
- `results/cs/exp*/summary_*.csv`: aggregate MAE/RMSE/R@1m/R@2m (± std).

If a teammate asks "why did sign X fail?", the per-sign CSV answers it.

---

## Done vs. still open

Done today:

- [x] All 11 aggregators + k-guard
- [x] All 5 experiments, validated
- [x] 3-level logging
- [x] Auto-generated tables + 6 figures
- [x] Filled the report §9–§12C tables with measured results

Open (next session, optional):

- [ ] A qualitative per-sign scatter (clean vs flagged points) like Figure 8, to
      show outlier detection visually.

---

## What to say if someone challenges the results

Straight from proposal §14; we are not overclaiming.

- *"Why doesn't L1-SOR beat NSAL on full data?"* Full clouds are already clean,
  and NSAL is strong there. Our methods target the sparse and outlier-corrupted
  regime (Exp 2, 3, 5), which is where they show up.
- *"Why does USPA barely help?"* The reliability cues are not perfectly
  calibrated to localization error. Honest, useful result.
- *"Why does MAE go up at k=20 in Exp 2?"* Different k means a different sign
  subset. Compare within a row, not across rows.

---

## Data provenance and honesty (read before presenting)

Not every artifact in the project is a real-world measured run, and that is
correct academic practice as long as we label each kind honestly. The integrity
risk is never fabricated data; it is mislabeling. Here is exactly what each thing
is.

| Artifact | What it actually is |
|---|---|
| Exp 1–4 numbers, all CSVs, the 6 figures | Real run of our code over real KITTI-TS data. Deterministic: re-running reproduces the saved CSVs to the digit. Figures are rendered from those CSVs, not hand-drawn. |
| Proposal §9–§12 filled tables | Transcribed from those CSVs, values checked to match. |
| Exp 5 phase-transition heatmaps | Synthetic by design. The data is generated, not measured. |
| Input location points | Real run, but derived through a fixed camera model and assumed sign heights, not ground-truth positions. |
| Pipeline diagrams and the ASCII sketch | Hand-authored illustrations, not generated output. |
| Rows marked "(paper)" / "Paper Reference" | Transcribed from Han et al. 2025, not our runs. |

Four things to say out loud so nobody is misled:

1. Exp 5 uses synthetic data on purpose. Compressed-sensing theory needs a known
   ground-truth outlier set to measure exact recovery against, and real data has
   none. So we generate a true location plus injected outliers and recover them.
   Say "synthetic" when you show those heatmaps; never imply they are real signs.
2. Our location points carry modeling assumptions. They come from KITTI camera
   constants (focal length 721.5377 px, center 621 px, FOV 45 degrees) and a
   table of German-standard sign heights by category, not from per-sign
   measurement. "Recovered true location" means recovered relative to that model.
3. The diagrams are conceptual. We drew the pipeline figure and the ASCII sketch
   by hand to explain the flow. They are illustrations, not results.
4. The paper-comparison numbers are the paper's. Any row labeled "(paper)" is
   from Han et al. 2025, cited, not produced by us.

Scope of this session's verification: we re-ran and confirmed only the CS
contribution in `AutoTS/sparse_localization/`. We did not re-execute the earlier
reproduction pipeline (Table I–IV, Figure 7/8) this session. Those numbers come
from prior runs documented in `reproduce_readme.md`. The "`nsal` here = 2.36
matches Table II" cross-check is genuine, but the Table II value it matches came
from that earlier reproduction, not from a run we repeated today.

---

*Generated 2026-06-23. Module: `AutoTS/sparse_localization/`. Proposal:
`2026_06_18_compressed_sensing_autots_proposal_report_18.md`.*
