"""
runner.py — shared evaluation + 3-level logging (section 12B)
============================================================
Runs an aggregator over a list of SignRecords and emits the per-sign rows the
proposal asks for (est/gt/err, outlier support + scores, residual, fallback
flag, runtime). Experiments build the run/summary levels on top of this.
"""

from __future__ import annotations

import csv
import time

import numpy as np

from . import aggregators as agg
from . import metrics
from .data import center_to_latlon

PER_SIGN_FIELDS = [
    "sign_id", "method", "k_used", "seed",
    "est_lat", "est_lon", "gt_lat", "gt_lon", "err_m", "hit_1m", "hit_2m",
    "n_outliers_detected", "outlier_idx", "residual_norm", "n_iter",
    "assumed_s", "fallback_triggered", "runtime_ms",
]


def eval_record(method, rec, coords=None, meta=None, seed=0, **kwargs):
    """Aggregate one sign and return (error_m, per_sign_row)."""
    coords = rec.coords if coords is None else coords
    meta = rec.meta if meta is None else meta
    k = len(coords)

    t0 = time.perf_counter()
    res = agg.aggregate(method, coords, meta, seed=seed, **kwargs)
    runtime_ms = (time.perf_counter() - t0) * 1000

    pred = center_to_latlon(res.center)
    err = metrics.error_m(pred, rec.gt)
    scores = res.outlier_scores
    row = {
        "sign_id": rec.sign_id,
        "method": method,
        "k_used": k,
        "seed": seed,
        "est_lat": pred[0], "est_lon": pred[1],
        "gt_lat": rec.gt[0], "gt_lon": rec.gt[1],
        "err_m": err,
        "hit_1m": int(err < 1.0), "hit_2m": int(err < 2.0),
        "n_outliers_detected": len(res.outlier_idx),
        "outlier_idx": "|".join(map(str, res.outlier_idx)),
        "residual_norm": "" if res.residual_norm is None else round(res.residual_norm, 6),
        "n_iter": "" if res.n_iter is None else res.n_iter,
        "assumed_s": "" if res.assumed_s is None else res.assumed_s,
        "fallback_triggered": int(res.fallback_triggered),
        "runtime_ms": round(runtime_ms, 4),
    }
    return err, row


def run_method(method, records, seed=0, **kwargs):
    """Evaluate a method over all records -> (summary dict, per_sign rows)."""
    errors, rows = [], []
    for rec in records:
        err, row = eval_record(method, rec, seed=seed, **kwargs)
        errors.append(err)
        rows.append(row)
    summ = metrics.summarize(errors)
    summ["fallback_rate"] = float(np.mean([r["fallback_triggered"] for r in rows])) \
        if rows else 0.0
    return summ, rows


def write_per_sign(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_SIGN_FIELDS)
        w.writeheader()
        w.writerows(rows)


def write_summary(path, summary_rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)
