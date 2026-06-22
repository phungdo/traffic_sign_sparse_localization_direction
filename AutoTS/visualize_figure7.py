#!/usr/bin/env python3
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from orientation_repro_lib import ensure_dir, safe_torch_load, split_cache_path


def find_sample(payload, sign_id):
    for sample in payload["samples"]:
        if str(sample["sign_id"]) == str(sign_id):
            return sample
    raise KeyError(f"sign id {sign_id} not found in cache")


def central_12(feature_list):
    arr = np.stack([x.numpy() for x in feature_list], axis=0)
    if arr.shape[0] >= 12:
        start = (arr.shape[0] - 12) // 2
        return arr[start:start + 12], list(range(start, start + 12)), 0
    pad = np.zeros((12, arr.shape[1]), dtype=arr.dtype)
    pad[:arr.shape[0]] = arr
    return pad, list(range(arr.shape[0])), 12 - arr.shape[0]


def main():
    parser = argparse.ArgumentParser(description="Generate paper-style Figure 7 SIFT heatmaps.")
    parser.add_argument("--cache-dir", default="./results/orientation_cache")
    parser.add_argument("--backward-id", default="113")
    parser.add_argument("--rightward-id", default="288")
    parser.add_argument("--out", default="./results/figure7/figure7_sift_heatmaps.png")
    parser.add_argument("--limit", type=int, default=None,
                        help="Read a limit-specific cache for smoke tests.")
    args = parser.parse_args()

    cache_path = split_cache_path(args.cache_dir, "test", args.limit)
    payload = safe_torch_load(cache_path, map_location="cpu")
    backward = find_sample(payload, args.backward_id)
    rightward = find_sample(payload, args.rightward_id)

    back_arr, back_rows, back_padding = central_12(backward["sift_feature"])
    right_arr, right_rows, right_padding = central_12(rightward["sift_feature"])

    ensure_dir(os.path.dirname(args.out))
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 4.4), dpi=180)
    for ax, arr, title in [
        (axes[0], back_arr, "Position-aware SIFT features of backward traffic sign"),
        (axes[1], right_arr, "Position-aware SIFT features of rightward traffic sign"),
    ]:
        ax.imshow(arr, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_ylabel("Traffic sign sequence", fontsize=10)
        ax.set_yticks(np.arange(12))
        ax.set_yticklabels(np.arange(1, 13), fontsize=7)
        ax.set_xticks([])
        ax.set_xlabel(title, fontsize=11)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout(h_pad=1.4)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "output": args.out,
        "cache": cache_path,
        "backward_id": str(args.backward_id),
        "rightward_id": str(args.rightward_id),
        "backward_original_rows": len(backward["sift_feature"]),
        "rightward_original_rows": len(rightward["sift_feature"]),
        "rendered_rows": 12,
        "feature_dim": int(back_arr.shape[1]),
        "backward_selected_indices": back_rows,
        "rightward_selected_indices": right_rows,
        "backward_padded_rows": back_padding,
        "rightward_padded_rows": right_padding,
    }
    with open(os.path.splitext(args.out)[0] + "_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved Figure 7 to {args.out}")


if __name__ == "__main__":
    main()
