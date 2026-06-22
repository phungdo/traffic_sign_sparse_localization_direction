"""
run_all.py — run every CS experiment and log a reproducibility manifest
=======================================================================
    python -m sparse_localization.run_all              # all experiments
    python -m sparse_localization.run_all 1 4 5        # only the listed ones

Writes a per-run manifest to AutoTS/results/cs/runs.csv (proposal section
12B.1) recording which experiment ran, when, and the key hyperparameters, so
the whole study can be re-derived from one file.
"""

from __future__ import annotations

import csv
import os
import sys
import time

from . import aggregators as agg
from .experiments import (exp1_main_table, exp2_sparse_k, exp3_outlier_stress,
                          exp4_uspa_ablation, exp5_synthetic_phase)

OUT = "./results/cs"
EXPERIMENTS = {
    "1": ("main_table", exp1_main_table.main),
    "2": ("sparse_k", exp2_sparse_k.main),
    "3": ("outlier_stress", exp3_outlier_stress.main),
    "4": ("uspa_ablation", exp4_uspa_ablation.main),
    "5": ("synthetic_phase", exp5_synthetic_phase.main),
}

# Hyperparameters in force across the study (logged for reproducibility).
HPARAMS = {
    "K_min": agg.K_MIN,
    "l1_lambda": 2.0,
    "nsal_sigma": 2.5,
    "uspa_lambda_d": 0.02,
    "greedy_sensing": "gaussian",
    "exp2_seeds": "0-4",
    "exp3_seeds": "0-2",
    "exp3_magnitudes": "5,10,20",
    "exp5_trials": 100,
}


def main():
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1:] or list(EXPERIMENTS)
    runs = []
    for key in which:
        name, fn = EXPERIMENTS[key]
        print("\n" + "=" * 70 + f"\nExperiment {key}: {name}\n" + "=" * 70)
        t0 = time.time()
        fn()
        runs.append({
            "run_id": f"exp{key}_{int(t0)}",
            "experiment": name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
            "runtime_s": round(time.time() - t0, 1),
            **HPARAMS,
        })

    manifest = os.path.join(OUT, "runs.csv")
    write_header = not os.path.exists(manifest)
    with open(manifest, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(runs)
    print(f"\nLogged {len(runs)} run(s) -> {manifest}")


if __name__ == "__main__":
    main()
