"""
Experiment 2 — Controlled sparse observation benchmark (proposal section 10)
============================================================================
For k in {2,3,5,10,20,all} we subsample k points per sign (multiple seeds) and
measure how each method degrades. This is the core COMP5340 question: how well
can we recover the signal from few measurements? The k-guard makes the sparse
methods fall back to the geometric median for k < 7 (expected, not a bug).

    python -m sparse_localization.experiments.exp2_sparse_k

Outputs (AutoTS/results/cs/exp2/):
    summary_exp2.csv     (method, k) -> MAE/RMSE/R@1m/R@2m mean+/-std, fallback_rate
    mae_table_exp2.csv   pivot: rows = k, cols = method (MAE mean)
"""

from __future__ import annotations

import csv
import os

import numpy as np

from .. import data, metrics
from ..runner import eval_record

OUT = "./results/cs/exp2"
K_VALUES = [2, 3, 5, 10, 20, "all"]
METHODS = ["mean", "median", "geometric_median", "dbscan", "nsal",
           "l1_sor", "omp", "cosamp", "sp", "uspa"]
SEEDS = [0, 1, 2, 3, 4]


def main():
    os.makedirs(OUT, exist_ok=True)
    records = data.load_records(depth_mode="planedepth")
    print(f"Loaded {len(records)} signs\n")

    summary_rows = []
    pivot = {}  # method -> {k: mae_mean}

    for kv in K_VALUES:
        for method in METHODS:
            seed_maes, seed_rmse, seed_r1, seed_r2, fb = [], [], [], [], []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                errors, fb_flags = [], []
                for rec in records:
                    if kv != "all" and rec.k < kv:
                        continue  # sign cannot supply k points
                    coords, meta = (rec.coords, rec.meta) if kv == "all" \
                        else data.subsample(rec, kv, rng)
                    err, row = eval_record(method, rec, coords=coords,
                                           meta=meta, seed=seed)
                    errors.append(err)
                    fb_flags.append(row["fallback_triggered"])
                s = metrics.summarize(errors)
                seed_maes.append(s["mae"]); seed_rmse.append(s["rmse"])
                seed_r1.append(s["r1m"]); seed_r2.append(s["r2m"])
                fb.append(np.mean(fb_flags) if fb_flags else 0.0)

            mae_mean = float(np.mean(seed_maes))
            summary_rows.append({
                "k": kv, "method": method,
                "mae_mean": round(mae_mean, 3),
                "mae_std": round(float(np.std(seed_maes)), 3),
                "rmse_mean": round(float(np.mean(seed_rmse)), 3),
                "r1m_mean": round(float(np.mean(seed_r1)), 2),
                "r2m_mean": round(float(np.mean(seed_r2)), 2),
                "fallback_rate": round(float(np.mean(fb)), 3),
                "n_signs": s["n"],
            })
            pivot.setdefault(method, {})[kv] = mae_mean
        print(f"k={str(kv):>3}  done")

    with open(os.path.join(OUT, "summary_exp2.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)

    # MAE pivot table (rows = k, cols = method)
    with open(os.path.join(OUT, "mae_table_exp2.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k"] + METHODS)
        for kv in K_VALUES:
            w.writerow([kv] + [round(pivot[m][kv], 3) for m in METHODS])

    # Console pivot
    print("\nMAE (mean over seeds) by k:")
    print("k    " + "".join(f"{m[:7]:>9}" for m in METHODS))
    for kv in K_VALUES:
        print(f"{str(kv):>4} " + "".join(f"{pivot[m][kv]:>9.3f}" for m in METHODS))
    print(f"\nSaved -> {OUT}/summary_exp2.csv, mae_table_exp2.csv")


if __name__ == "__main__":
    main()
