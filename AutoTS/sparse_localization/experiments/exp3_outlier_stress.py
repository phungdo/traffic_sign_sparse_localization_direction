"""
Experiment 3 — Sparse outlier stress test (proposal section 11)
===============================================================
Inject artificial gross outliers into each sign's point set and measure how
fast each method degrades. This directly tests the sparse-outlier assumption
p_i = l_s + e_i + eps_i: a true sparse-recovery method should tolerate a few
large corruptions as long as the majority of observations stay clean.

    python -m sparse_localization.experiments.exp3_outlier_stress

Outputs (AutoTS/results/cs/exp3/):
    summary_exp3.csv     (magnitude, ratio, method) -> MAE/RMSE/R@1m/R@2m + outlier-detection P/R
    mae_table_exp3.csv   pivot at the headline magnitude: rows = ratio, cols = method
"""

from __future__ import annotations

import csv
import os

import numpy as np

from .. import aggregators as agg
from .. import data, metrics
from ..runner import eval_record

OUT = "./results/cs/exp3"
RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4]
MAGNITUDES = [5.0, 10.0, 20.0]
HEADLINE_MAG = 10.0
METHODS = ["mean", "median", "geometric_median", "dbscan", "nsal",
           "l1_sor", "omp", "cosamp", "sp", "uspa"]
SEEDS = [0, 1, 2]
MIN_K = 7  # only stress signs with enough points to define an inlier majority


def detection_pr(pred_idx, true_mask):
    """Precision/recall of flagged outliers vs the injected ground truth."""
    pred = set(pred_idx)
    true = set(np.where(true_mask)[0].tolist())
    if not pred and not true:
        return 1.0, 1.0
    tp = len(pred & true)
    prec = tp / len(pred) if pred else (1.0 if not true else 0.0)
    rec = tp / len(true) if true else 1.0
    return prec, rec


def main():
    os.makedirs(OUT, exist_ok=True)
    records = [r for r in data.load_records(depth_mode="planedepth") if r.k >= MIN_K]
    print(f"Loaded {len(records)} signs with k>={MIN_K}\n")

    summary_rows = []
    pivot = {}  # method -> {ratio: mae} at HEADLINE_MAG

    for mag in MAGNITUDES:
        for ratio in RATIOS:
            for method in METHODS:
                maes, precs, recs = [], [], []
                for seed in SEEDS:
                    rng = np.random.default_rng(seed)
                    errors, p_list, r_list = [], [], []
                    for rec in records:
                        coords, mask = data.inject_outliers(rec.coords, ratio, mag, rng)
                        err, row = eval_record(method, rec, coords=coords,
                                               meta=rec.meta, seed=seed)
                        errors.append(err)
                        if method in agg.SPARSE_METHODS and mask.any():
                            idx = [int(i) for i in row["outlier_idx"].split("|") if i != ""]
                            p, r = detection_pr(idx, mask)
                            p_list.append(p); r_list.append(r)
                    maes.append(metrics.summarize(errors)["mae"])
                    if p_list:
                        precs.append(np.mean(p_list)); recs.append(np.mean(r_list))
                mae_mean = float(np.mean(maes))
                summary_rows.append({
                    "magnitude": mag, "ratio": ratio, "method": method,
                    "mae_mean": round(mae_mean, 3),
                    "mae_std": round(float(np.std(maes)), 3),
                    "detect_precision": round(float(np.mean(precs)), 3) if precs else "",
                    "detect_recall": round(float(np.mean(recs)), 3) if recs else "",
                })
                if mag == HEADLINE_MAG:
                    pivot.setdefault(method, {})[ratio] = mae_mean
        print(f"magnitude={mag:>4}m done")

    with open(os.path.join(OUT, "summary_exp3.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)

    with open(os.path.join(OUT, "mae_table_exp3.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"ratio (mag={HEADLINE_MAG}m)"] + METHODS)
        for ratio in RATIOS:
            w.writerow([ratio] + [round(pivot[m][ratio], 3) for m in METHODS])

    print(f"\nMAE vs outlier ratio (magnitude {HEADLINE_MAG}m):")
    print("ratio" + "".join(f"{m[:7]:>9}" for m in METHODS))
    for ratio in RATIOS:
        print(f"{ratio:>5}" + "".join(f"{pivot[m][ratio]:>9.3f}" for m in METHODS))
    print(f"\nSaved -> {OUT}/summary_exp3.csv, mae_table_exp3.csv")


if __name__ == "__main__":
    main()
