"""
Experiment 5 — Synthetic recovery / phase transition (proposal section 12C)
===========================================================================
Real data has no ground-truth outlier support, so we generate it: fix a true
location l, draw m observations p_i = l + eps_i, then inject s gross outliers at
a random support. We recover the support with each CS method and measure how
often it is recovered exactly as a function of (m, s). This is the direct
check that the methods behave the way compressed-sensing theory predicts.

    python -m sparse_localization.experiments.exp5_synthetic_phase

Outputs (AutoTS/results/cs/exp5/):
    summary_exp5.csv          (method, m, s) -> exact-recovery rate, P/R, l_err
    phase_<method>.csv        heatmap matrix: rows = s, cols = m (exact-recovery)
"""

from __future__ import annotations

import csv
import os

import numpy as np

from .. import aggregators as agg

OUT = "./results/cs/exp5"
M_VALUES = [4, 8, 12, 16]
S_VALUES = [1, 2, 3]
N_TRIALS = 100
NOISE_STD = 0.3
OUTLIER_MAG = 8.0
L1_LAMBDA = 1.5
METHODS = ["omp", "cosamp", "sp", "l1_sor"]


def make_trial(m, s, rng):
    """Return (coords, true_support, true_l)."""
    true_l = rng.normal(0, 5, size=2)
    coords = true_l + rng.normal(0, NOISE_STD, size=(m, 2))
    support = set(rng.choice(m, size=min(s, m), replace=False).tolist())
    for i in support:
        theta = rng.uniform(0, 2 * np.pi)
        coords[i] += OUTLIER_MAG * np.array([np.cos(theta), np.sin(theta)])
    return coords, support, true_l


def recover_support(method, coords, s):
    """Recover (outlier support, estimated location) for a CS method."""
    if method == "l1_sor":
        res = agg.l1_sor(coords, lam=L1_LAMBDA)
        return set(res.outlier_idx), res.center
    support, e_hat, _ = agg._greedy_support(coords, s, method,
                                             sensing="gaussian", seed=0)
    inliers = [i for i in range(len(coords)) if i not in support]
    center = coords[inliers].mean(axis=0) if inliers else coords.mean(axis=0)
    return support, center


def main():
    os.makedirs(OUT, exist_ok=True)
    summary_rows = []
    phase = {m: {} for m in METHODS}  # method -> {(s,m): rate}

    for method in METHODS:
        for s in S_VALUES:
            for m in M_VALUES:
                exact, precs, recs, lerrs = [], [], [], []
                for t in range(N_TRIALS):
                    rng = np.random.default_rng(1000 * s + 10 * m + t)
                    coords, true_sup, true_l = make_trial(m, s, rng)
                    pred_sup, est = recover_support(method, coords, s)
                    exact.append(int(pred_sup == true_sup))
                    tp = len(pred_sup & true_sup)
                    precs.append(tp / len(pred_sup) if pred_sup else (1.0 if not true_sup else 0.0))
                    recs.append(tp / len(true_sup) if true_sup else 1.0)
                    lerrs.append(float(np.linalg.norm(est - true_l)))
                rate = float(np.mean(exact))
                phase[method][(s, m)] = rate
                summary_rows.append({
                    "method": method, "m": m, "s": s,
                    "exact_recovery": round(rate, 3),
                    "support_precision": round(float(np.mean(precs)), 3),
                    "support_recall": round(float(np.mean(recs)), 3),
                    "l_err_mean": round(float(np.mean(lerrs)), 4),
                })
        print(f"{method:<8} done")

    with open(os.path.join(OUT, "summary_exp5.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)

    for method in METHODS:
        with open(os.path.join(OUT, f"phase_{method}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["s\\m"] + M_VALUES)
            for s in S_VALUES:
                w.writerow([s] + [phase[method][(s, m)] for m in M_VALUES])

    print("\nExact-recovery rate (rows s, cols m):")
    for method in METHODS:
        print(f"\n  {method}")
        print("   s\\m " + "".join(f"{m:>7}" for m in M_VALUES))
        for s in S_VALUES:
            print(f"   {s:>3} " + "".join(f"{phase[method][(s, m)]:>7.2f}" for m in M_VALUES))
    print(f"\nSaved -> {OUT}/summary_exp5.csv, phase_<method>.csv")


if __name__ == "__main__":
    main()
