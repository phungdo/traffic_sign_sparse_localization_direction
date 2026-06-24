"""augment_map_methods.py — add per-frame points + per-method locations
========================================================================
Enriches an existing ``results/map/map_sign_results.json`` with two new fields
per sign (``points`` and ``methods``) WITHOUT re-running the detector or the
orientation model. It reads the per-frame location points straight from the
orientation cache, so it runs on CPU on any machine.

    python augment_map_methods.py
    python augment_map_methods.py --map results/map/map_sign_results.json \
                                  --cache-dir results/orientation_cache

Use this when you already have a map JSON (built on a GPU box) and just want the
new per-method / per-image data. A full rebuild via ``build_map_sign_results.py``
now also writes these fields.
"""

from __future__ import annotations

import argparse
import json
import os

from orientation_repro_lib import safe_torch_load
from map_methods import MAP_METHODS, sign_method_data


def load_cache_points(cache_dir):
    """Return ``{"<split>/<sign_id>": (location_point, geolocation, frame_ids)}``."""
    index = {}
    for split in ("train", "test"):
        path = os.path.join(cache_dir, f"{split}_features.pt")
        if not os.path.exists(path):
            print(f"[augment] cache missing: {path} (skipped)")
            continue
        payload = safe_torch_load(path)
        samples = payload["samples"] if isinstance(payload, dict) else payload
        for s in samples:
            key = f"{s['split']}/{s['sign_id']}"
            index[key] = (
                s.get("location_point", []),
                [float(x) for x in s.get("geolocation", [])],
                list(s.get("frame_ids", [])),
            )
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="./results/map/map_sign_results.json")
    parser.add_argument("--cache-dir", default="./results/orientation_cache")
    args = parser.parse_args()

    with open(args.map) as f:
        data = json.load(f)
    signs = data["signs"]

    cache = load_cache_points(args.cache_dir)

    n_done = n_nocache = 0
    max_nsal_gap = 0.0
    for key, sign in signs.items():
        entry = cache.get(key)
        if entry is None:
            n_nocache += 1
            continue
        points, gt, frame_ids = entry
        md = sign_method_data(points, gt, frame_ids)
        sign["points"] = md["points"]
        sign["methods"] = md["methods"]
        # Sanity: the new nsal location should match the existing AutoTS est.
        if sign.get("est_lat") is not None and "nsal" in md["methods"]:
            gap = abs(md["methods"]["nsal"]["est_lat"] - sign["est_lat"]) \
                + abs(md["methods"]["nsal"]["est_lon"] - sign["est_lon"])
            max_nsal_gap = max(max_nsal_gap, gap)
        n_done += 1

    data["map_methods"] = MAP_METHODS  # selector order for the UI
    with open(args.map, "w") as f:
        json.dump(data, f)

    size_mb = os.path.getsize(args.map) / 1e6
    print(f"AUGMENTED {args.map}")
    print(f"  signs enriched : {n_done}  (no cache match: {n_nocache})")
    print(f"  methods/sign   : {len(MAP_METHODS)}  -> {MAP_METHODS}")
    print(f"  nsal vs existing AutoTS max |dlat|+|dlon|: {max_nsal_gap:.2e} (≈0 expected)")
    print(f"  file size      : {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
