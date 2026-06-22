"""
Experiment 4 — USPA reliability ablation (proposal section 12)
==============================================================
USPA only helps if the reliability cues (depth, bbox area, confidence) actually
separate good observations from bad ones. We sweep the q_i variants and report
localization metrics. 'none' (q_i=1) reduces USPA to a degree-weighted center.

    python -m sparse_localization.experiments.exp4_uspa_ablation

Output (AutoTS/results/cs/exp4/):
    summary_exp4.csv   one row per reliability variant (+ NSAL reference)
"""

from __future__ import annotations

import csv
import os

from .. import data, runner

OUT = "./results/cs/exp4"
VARIANTS = [
    ("none", "q_i = 1 (degree-weighted center)"),
    ("depth", "q_i = exp(-lambda_d * depth)"),
    ("area", "q_i = normalized bbox area"),
    ("area_depth", "q_i = area * exp(-lambda_d * depth)"),
    ("conf_area_depth", "q_i = conf * area * exp(-lambda_d * depth)"),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    records = data.load_records(depth_mode="planedepth")
    print(f"Loaded {len(records)} signs\n")

    rows = []
    # NSAL reference (full paper method, includes MinCut) for context.
    summ, _ = runner.run_method("nsal", records)
    rows.append({"cue": "NSAL (reference)", "formula": "graph density + MinCut",
                 "mae": round(summ["mae"], 3), "rmse": round(summ["rmse"], 3),
                 "r1m": round(summ["r1m"], 2), "r2m": round(summ["r2m"], 2)})

    print(f"{'cue':<22}{'MAE':>8}{'RMSE':>8}{'R@1m':>8}{'R@2m':>8}")
    print("-" * 54)
    print(f"{'NSAL (reference)':<22}{summ['mae']:>8.3f}{summ['rmse']:>8.3f}"
          f"{summ['r1m']:>8.2f}{summ['r2m']:>8.2f}")

    for variant, formula in VARIANTS:
        summ, _ = runner.run_method("uspa", records, variant=variant)
        rows.append({"cue": variant, "formula": formula,
                     "mae": round(summ["mae"], 3), "rmse": round(summ["rmse"], 3),
                     "r1m": round(summ["r1m"], 2), "r2m": round(summ["r2m"], 2)})
        print(f"{variant:<22}{summ['mae']:>8.3f}{summ['rmse']:>8.3f}"
              f"{summ['r1m']:>8.2f}{summ['r2m']:>8.2f}")

    with open(os.path.join(OUT, "summary_exp4.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cue", "formula", "mae", "rmse", "r1m", "r2m"])
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved -> {OUT}/summary_exp4.csv")


if __name__ == "__main__":
    main()
