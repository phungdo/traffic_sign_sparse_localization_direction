"""map_methods.py — per-frame points + per-method locations for the map popup
=============================================================================
Pure-geometry helper (CPU only, no detector / orientation model needed).

For one sign it turns the per-frame location points + GT into:
  - ``points``  : the per-*image* predicted location dots (method-independent
                  input — one per frame), so the popup can show every frame's
                  raw estimate.
  - ``methods`` : each aggregator's single recovered location + error vs GT
                  (the per-*method* output), so the popup can switch methods.

It reuses the *validated* EPSG:3044 projection from ``orientation_repro_lib``
(the same one that produced the existing ``est_lat/est_lon``) and the COMP5340
aggregators from ``sparse_localization.aggregators``. Because the projection and
NSAL implementation match, the ``nsal`` method here reproduces the map's existing
AutoTS location (a built-in sanity check).
"""

from __future__ import annotations

import numpy as np
from geopy.distance import geodesic

from orientation_repro_lib import (
    init_projection_transforms,
    point_transform,
    point_transform_back,
)
from sparse_localization import aggregators as agg

# Methods exposed in the popup selector (same set/order as make_report).
MAP_METHODS = [
    "mean", "median", "geometric_median", "dbscan", "nsal",
    "l1_sor", "omp", "cosamp", "sp", "uspa",
]

# Friendly labels for the UI dropdown.
METHOD_LABELS = {
    "mean": "Mean",
    "median": "Median",
    "geometric_median": "Geometric median",
    "dbscan": "DBSCAN",
    "nsal": "AutoTS NSAL (paper)",
    "l1_sor": "L1-SOR (CS convex)",
    "omp": "OMP (CS greedy)",
    "cosamp": "CoSaMP (CS greedy)",
    "sp": "Subspace Pursuit (CS greedy)",
    "uspa": "USPA (proposed)",
}


def sign_method_data(points_latlon, gt_latlon, frame_ids=None):
    """Return ``{"points": [...], "methods": {...}}`` for one sign.

    ``points_latlon`` : list of ``[lat, lon]`` per-frame location points.
    ``gt_latlon``     : ``[lat, lon]`` ground-truth sign location (or None).
    ``frame_ids``     : optional list aligned with ``points_latlon``.
    """
    if not points_latlon:
        return {"points": [], "methods": {}}

    init_projection_transforms()
    # Project to the metric CRS exactly as the validated pipeline does.
    coords = np.array(
        [[p.x, p.y] for p in (point_transform(pt) for pt in points_latlon)],
        dtype=float,
    )

    has_gt = gt_latlon is not None and len(gt_latlon) >= 2

    # Per-image dots (method-independent input).
    points = []
    for i, pt in enumerate(points_latlon):
        points.append({
            "lat": float(pt[0]),
            "lon": float(pt[1]),
            "frame_id": (frame_ids[i] if frame_ids and i < len(frame_ids) else None),
        })

    # Per-method recovered location + error.
    methods = {}
    for m in MAP_METHODS:
        res = agg.aggregate(m, coords)
        back = point_transform_back(res.center)
        est = [float(back.x), float(back.y)]
        err = geodesic(est, gt_latlon).meters if has_gt else None
        methods[m] = {
            "label": METHOD_LABELS[m],
            "est_lat": est[0],
            "est_lon": est[1],
            "error_m": round(float(err), 2) if err is not None else None,
            "outlier_idx": [int(i) for i in res.outlier_idx],
            "fallback": bool(res.fallback_triggered),
        }
    return {"points": points, "methods": methods}
