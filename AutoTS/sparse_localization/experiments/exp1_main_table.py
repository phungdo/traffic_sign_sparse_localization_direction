"""
Experiment 1 — Main localization table (proposal section 9)
===========================================================
Runs every method on the full per-sign point sets and reports
MAE / RMSE / R@1m / R@2m, plus per-sign logs for the sparse methods.

    python -m sparse_localization.experiments.exp1_main_table

Outputs (under AutoTS/results/cs/exp1/):
    summary_exp1.csv              one row per method
    per_sign_<method>.csv         per-sign diagnostics for each method
"""

from __future__ import annotations

import os

from .. import aggregators as agg
from .. import data, runner

OUT = "./results/cs/exp1"


def main():
    os.makedirs(OUT, exist_ok=True)
    records = data.load_records(depth_mode="planedepth")
    print(f"Loaded {len(records)} signs "
          f"(median k = {sorted(r.k for r in records)[len(records)//2]})")

    summary_rows = []
    print(f"\n{'Method':<18}{'MAE':>8}{'RMSE':>8}{'R@1m':>8}{'R@2m':>8}"
          f"{'N':>6}{'fb%':>7}")
    print("-" * 63)
    for method in agg.ALL_METHODS:
        summ, rows = runner.run_method(method, records)
        runner.write_per_sign(os.path.join(OUT, f"per_sign_{method}.csv"), rows)
        summary_rows.append({
            "method": method,
            "mae": round(summ["mae"], 3),
            "rmse": round(summ["rmse"], 3),
            "r1m": round(summ["r1m"], 2),
            "r2m": round(summ["r2m"], 2),
            "n": summ["n"],
            "fallback_rate": round(summ["fallback_rate"], 4),
        })
        print(f"{method:<18}{summ['mae']:>8.3f}{summ['rmse']:>8.3f}"
              f"{summ['r1m']:>8.2f}{summ['r2m']:>8.2f}{summ['n']:>6}"
              f"{summ['fallback_rate']*100:>7.1f}")

    runner.write_summary(
        os.path.join(OUT, "summary_exp1.csv"), summary_rows,
        ["method", "mae", "rmse", "r1m", "r2m", "n", "fallback_rate"])
    print(f"\nSaved -> {OUT}/summary_exp1.csv and per_sign_<method>.csv")


if __name__ == "__main__":
    main()
