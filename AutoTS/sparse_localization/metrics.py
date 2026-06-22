"""metrics.py — localization error metrics (section 7)."""

from __future__ import annotations

import numpy as np
from geopy.distance import geodesic


def error_m(pred_latlon, gt_latlon) -> float:
    """Per-sign Euclidean error in meters (geodesic), matching Table II."""
    return geodesic(pred_latlon, gt_latlon).meters


def summarize(errors) -> dict:
    """MAE, RMSE, Recall@1m, Recall@2m over a list of per-sign errors (m)."""
    e = np.asarray(errors, float)
    if e.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"),
                "r1m": 0.0, "r2m": 0.0, "n": 0}
    return {
        "mae": float(e.mean()),
        "rmse": float(np.sqrt((e ** 2).mean())),
        "r1m": float((e < 1.0).mean() * 100),
        "r2m": float((e < 2.0).mean() * 100),
        "n": int(e.size),
    }
