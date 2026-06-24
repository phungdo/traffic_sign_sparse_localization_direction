"""Generate per-sign Figure-8-style results for the KITTI-TS map popup.

For every cached sign it computes:
  - predicted orientation (from the retrained AutoTS orientation checkpoint),
  - AutoTS estimated location + absolute error vs GT GPS (NSAL full),
  - the same with MinCut disabled (no_mincut), for comparison,
  - a small JPEG thumbnail strip of the sign's frame images.

Output: results/map/map_sign_results.json, keyed by "<split>/<sign_id>".
"""
import argparse
import base64
import io
import json
import os

import numpy as np
import torch
from PIL import Image

from orientation_repro_lib import (
    load_cfg,
    safe_torch_load,
    localization_result,
    init_projection_transforms,
    ORIENTATION_NAMES,
    OrientationAblationNet,
)
from visualize_figure8 import load_samples, predict_sample
from map_methods import MAP_METHODS, sign_method_data


def frame_strip_b64(sample, img_root, max_frames=4, width=150, quality=62):
    """Return a list of base64 JPEG thumbnails sampled across the sign's frames."""
    frame_ids = list(sample.get("frame_ids", []))
    if not frame_ids:
        return []
    if len(frame_ids) > max_frames:
        idxs = np.linspace(0, len(frame_ids) - 1, max_frames).astype(int).tolist()
        frame_ids = [frame_ids[i] for i in idxs]
    thumbs = []
    for frame_id in frame_ids:
        path = os.path.join(img_root, f"{frame_id}.png")
        if not os.path.exists(path):
            continue
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        ratio = width / img.width
        img = img.resize((width, max(1, int(img.height * ratio))))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        thumbs.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return thumbs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", default="./orientation_config_v2_paper.yaml")
    parser.add_argument("--cache-dir", default="./results/orientation_cache")
    parser.add_argument("--orientation-ckpt",
                        default="./results/table4_orientation/AutoTS_ours_best.pth")
    parser.add_argument("--out", default="./results/map/map_sign_results.json")
    parser.add_argument("--max-frames", type=int, default=4)
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    _, all_samples = load_samples(args.cache_dir, None)

    model = OrientationAblationNet(cfg, encoder="transformer").to(cfg.MODEL.DEVICE)
    state = safe_torch_load(args.orientation_ckpt, map_location=cfg.MODEL.DEVICE)
    model.load_state_dict(state)
    model.eval()

    init_projection_transforms()
    roots = {"train": cfg.DATA.TRAIN_IMG_ROOT, "test": cfg.DATA.TEST_IMG_ROOT}

    results = {}
    n_ok = n_noloc = 0
    for sample in all_samples:
        split = sample["split"]
        sign_id = str(sample["sign_id"])
        key = f"{split}/{sign_id}"

        gt_dir = int(sample["direction"])
        pred_dir = int(predict_sample(cfg, model, sample))

        points = sample.get("location_point", [])
        true_loc = [float(x) for x in sample.get("geolocation", [])]
        has_gt = len(true_loc) >= 2
        can_loc = bool(points) and has_gt
        full = localization_result(points, true_loc, "full") if can_loc else None
        nomc = localization_result(points, true_loc, "no_mincut") if can_loc else None
        if full is None:
            n_noloc += 1
        else:
            n_ok += 1

        results[key] = {
            "split": split,
            "sign_id": sign_id,
            "gt_dir": gt_dir,
            "gt_dir_name": ORIENTATION_NAMES.get(gt_dir),
            "pred_dir": pred_dir,
            "pred_dir_name": ORIENTATION_NAMES.get(pred_dir),
            "orient_correct": bool(pred_dir == gt_dir),
            "gt_lat": true_loc[0] if has_gt else None,
            "gt_lon": true_loc[1] if has_gt else None,
            "n_points": len(points),
            "est_lat": float(full["pred"][0]) if full else None,
            "est_lon": float(full["pred"][1]) if full else None,
            "error_m": round(float(full["error_m"]), 2) if full else None,
            "error_nomincut_m": round(float(nomc["error_m"]), 2) if nomc else None,
            "seq_b64": frame_strip_b64(sample, roots[split], args.max_frames),
        }

        # Per-frame points + per-method recovered locations for the popup.
        md = sign_method_data(points, true_loc, sample.get("frame_ids"))
        results[key]["points"] = md["points"]
        results[key]["methods"] = md["methods"]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"signs": results, "map_methods": MAP_METHODS}, f)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"WROTE {args.out}  signs={len(results)} with_loc={n_ok} no_loc={n_noloc} "
          f"size={size_mb:.1f}MB")


if __name__ == "__main__":
    main()
