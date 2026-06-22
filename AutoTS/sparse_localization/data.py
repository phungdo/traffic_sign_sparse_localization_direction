"""
data.py — Per-sign location points + reliability metadata
=========================================================
Reuses the *validated* geometry from ``eval_table2_localization`` (camera
params, depth models, WGS84 -> EPSG:3044 projection) so the CS methods operate
in exactly the same coordinate space as the reproduced NSAL. For every sign we
return:

    points_latlon : list[[lat, lon]]              raw location points
    coords        : np.ndarray (k, 2)             projected metric coordinates
    meta          : {"depth": (k,), "area": (k,)} reliability cues for USPA
    gt            : [lat, lon]                     ground-truth sign location

``depth_mode`` selects the AutoTS point source ('planedepth', default) or the
GeoLocating thin-lens source ('thin_lens').
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

# Import the validated localization geometry. eval_table2_localization has no
# import-time side effects beyond building the pyproj transformers.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval_table2_localization as L  # noqa: E402


@dataclass
class SignRecord:
    sign_id: str
    points_latlon: list
    coords: np.ndarray          # (k, 2) projected metric coords (EPSG:3044)
    meta: dict                  # {"depth": (k,), "area": (k,)}
    gt: list                    # [lat, lon]

    @property
    def k(self) -> int:
        return len(self.coords)


def _points_with_meta(sign_data, boxes, depth_mode, img_depth):
    """Replicates L.compute_location_points but also keeps depth + bbox area."""
    image_yaws = sign_data.get("image_yaws", {})
    geolocs = sign_data.get("image_geolocations", {})
    category = sign_data["category"]
    if not image_yaws or not boxes:
        return [], [], []

    pts, depths, areas = [], [], []
    for obj_id, box in boxes.items():
        if depth_mode == "planedepth":
            depth_map = img_depth.get(obj_id) if img_depth else None
            if depth_map is None:
                continue
            depth = L.planedepth_depth(box, depth_map)
        else:
            depth = L.thin_lens_depth(box, category)
        if depth is None or depth <= 0:
            continue

        center_x = (box[0] + box[2]) / 2
        relative_angle = (center_x - L.CENTER_X) * (L.FOV_X / L.CENTER_X)
        yaw = np.degrees(image_yaws.get(obj_id, 0))
        alt = yaw - relative_angle
        lat0, lng0 = geolocs.get(obj_id, (None, None))
        if lat0 is None or lng0 is None:
            continue

        lat_, lng_ = L.get_geo(lat0, lng0, depth, alt)
        pts.append([lat_, lng_])
        depths.append(float(depth))
        areas.append(float((box[2] - box[0]) * (box[3] - box[1])))
    return pts, depths, areas


def load_records(depth_mode: str = "planedepth", min_points: int = 1):
    """Load every sign with at least ``min_points`` valid location points.

    ``planedepth`` needs ``./data/img_depth.npz`` (3.5 GB, not in the repo). If
    that file is absent we fall back to the thin-lens depth model, which needs
    only the sign GT JSONs, so the experiments run out of the box on a fresh
    clone. The numbers then shift slightly from the planedepth results saved
    under ``results/cs/`` (see the README setup notes).
    """
    if depth_mode == "planedepth" and not os.path.exists("./data/img_depth.npz"):
        print("[data] ./data/img_depth.npz not found; falling back to "
              "depth_mode='thin_lens' (see README setup).")
        depth_mode = "thin_lens"

    all_signs = L.load_all_signs()
    img_depth = None
    if depth_mode == "planedepth":
        npz = np.load("./data/img_depth.npz")
        img_depth = {key: npz[key] for key in npz.files}

    records = []
    for sign_id, sign_data in all_signs.items():
        gt = sign_data.get("Geolocation", [])
        if len(gt) < 2:
            continue
        boxes = sign_data.get("images", {})
        if not boxes:
            continue

        pts, depths, areas = _points_with_meta(sign_data, boxes, depth_mode, img_depth)
        if len(pts) < min_points:
            continue

        coords = np.array([[p.x, p.y] for p in
                           (L.point_transform(pt) for pt in pts)], float)
        records.append(SignRecord(
            sign_id=sign_id,
            points_latlon=pts,
            coords=coords,
            meta={"depth": np.array(depths), "area": np.array(areas)},
            gt=list(gt),
        ))
    return records


def center_to_latlon(center_xy):
    """Projected metric coords -> [lat, lon] (inverse of L.point_transform)."""
    back = L.point_transform_back(center_xy)
    return [back.x, back.y]


def subsample(rec: SignRecord, k: int, rng):
    """Return (coords, meta) for k randomly chosen observations of a sign.

    Used by the controlled-sparsity benchmark (Experiment 2). If the sign has
    fewer than k points, all of them are returned.
    """
    n = rec.k
    if k >= n:
        return rec.coords, rec.meta
    idx = rng.choice(n, size=k, replace=False)
    meta = {key: np.asarray(val)[idx] for key, val in rec.meta.items()}
    return rec.coords[idx], meta


def inject_outliers(coords, ratio, magnitude, rng):
    """Corrupt a fraction ``ratio`` of points by adding a random metric vector
    of norm ``magnitude`` (Experiment 3). Returns (coords', outlier_mask)."""
    coords = np.array(coords, float)
    n = len(coords)
    n_out = int(round(ratio * n))
    mask = np.zeros(n, dtype=bool)
    if n_out == 0:
        return coords, mask
    idx = rng.choice(n, size=n_out, replace=False)
    theta = rng.uniform(0, 2 * np.pi, size=n_out)
    delta = magnitude * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    coords[idx] += delta
    mask[idx] = True
    return coords, mask
