#!/usr/bin/env python3
import argparse
import csv
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

from orientation_repro_lib import (
    CachedOrientationDataset,
    GOOGLE_ROAD_TILE_URL,
    GOOGLE_SATELLITE_TILE_URL,
    OrientationAblationNet,
    ORIENTATION_NAMES,
    draw_boxed_sequence,
    ensure_dir,
    init_projection_transforms,
    load_cfg,
    localization_result,
    orientation_collate,
    orientation_panel,
    render_points_panel,
    safe_torch_load,
    split_cache_path,
)


PAPER_ROW4_TARGET = (48 + 59 / 60 + 0.8 / 3600, 8 + 24 / 60 + 6.9 / 3600)

PAPER_FIG8_ROWS = [
    {
        "split": "train",
        "sign_id": "3",
        "ablation": "AutoTS w/o MinCut",
        "paper_sign_label": "Yield",
        "paper_gt_direction": 1,
        "paper_pred_direction": 1,
        "paper_autots_error_m": 2.15,
        "paper_ablation_error_m": 3.39,
        "paper_caption_geolocation": [49 + 0 / 60 + 32.0 / 3600,
                                       8 + 26 / 60 + 22.0 / 3600],
    },
    {
        "split": "train",
        "sign_id": "274",
        "ablation": "AutoTS w/o MinCut",
        "paper_sign_label": "RoadWork",
        "paper_gt_direction": 1,
        "paper_pred_direction": 1,
        "paper_autots_error_m": 1.57,
        "paper_ablation_error_m": 1.57,
        "paper_caption_geolocation": [48 + 56 / 60 + 56.8 / 3600,
                                       8 + 29 / 60 + 19.3 / 3600],
    },
    {
        "split": "test",
        "sign_id": "2",
        "ablation": "AutoTS w/o Weight",
        "paper_sign_label": "Yield",
        "paper_gt_direction": 2,
        "paper_pred_direction": 2,
        "paper_autots_error_m": 0.85,
        "paper_ablation_error_m": 1.13,
        "paper_caption_geolocation": [49 + 0 / 60 + 39.2 / 3600,
                                       8 + 25 / 60 + 24.0 / 3600],
    },
    {
        "split": "train",
        "sign_id": "174",
        "ablation": "AutoTS w/o Weight",
        "paper_sign_label": "NoStay",
        "paper_gt_direction": 1,
        "paper_pred_direction": 2,
        "paper_autots_error_m": 0.81,
        "paper_ablation_error_m": 0.88,
        "paper_caption_geolocation": [PAPER_ROW4_TARGET[0], PAPER_ROW4_TARGET[1]],
    },
]


def load_samples(cache_dir, limit=None):
    samples = {}
    all_samples = []
    for split in ["train", "test"]:
        payload = safe_torch_load(split_cache_path(cache_dir, split, limit),
                                  map_location="cpu")
        for sample in payload["samples"]:
            key = (split, str(sample["sign_id"]))
            samples[key] = sample
            all_samples.append(sample)
    return samples, all_samples


def predict_sample(cfg, model, sample):
    dataset = CachedOrientationDataset([sample], "AutoTS (ours)", cfg.MODEL.DEVICE)
    batch = orientation_collate([dataset[0]])
    with torch.no_grad():
        logits = model(batch)
    return int(logits.argmax(dim=1).cpu().numpy()[0])


def sample_is_renderable(sample):
    return (
        len(sample.get("geolocation", [])) >= 2 and
        len(sample.get("location_point", [])) > 0
    )


def first_unused_sample(all_samples, used_keys):
    for sample in sorted(all_samples, key=lambda s: (s["split"], int(s["sign_id"]))):
        key = (sample["split"], str(sample["sign_id"]))
        if key not in used_keys and sample_is_renderable(sample):
            used_keys.add(key)
            return sample
    return None


def select_paper_sample(spec, samples, all_samples, used_keys):
    key = (spec["split"], spec["sign_id"])
    if key in samples and key not in used_keys and sample_is_renderable(samples[key]):
        used_keys.add(key)
        return samples[key], "paper specified sample"

    fallback = first_unused_sample(all_samples, used_keys)
    if fallback is None:
        raise ValueError(f"No renderable fallback sample for paper row {key}.")
    if key not in samples:
        reason = f"fallback because paper ID {spec['split']}/{spec['sign_id']} is missing"
    elif key in used_keys:
        reason = f"fallback because paper ID {spec['split']}/{spec['sign_id']} was already used"
    else:
        reason = f"fallback because paper ID {spec['split']}/{spec['sign_id']} has no GPS/location points"
    return fallback, reason


def build_paper_deviation(spec, sample, pred, full, alt):
    notes = []
    sample_key = (sample["split"], str(sample["sign_id"]))
    paper_key = (spec["split"], spec["sign_id"])
    if sample_key != paper_key:
        notes.append(f"uses fallback sample {sample_key[0]}/{sample_key[1]}")
    if int(sample["direction"]) != spec["paper_gt_direction"]:
        notes.append(
            "GT differs from paper "
            f"({ORIENTATION_NAMES.get(spec['paper_gt_direction'])})"
        )
    if pred != spec["paper_pred_direction"]:
        notes.append(
            "prediction differs from paper "
            f"({ORIENTATION_NAMES.get(spec['paper_pred_direction'])})"
        )
    if abs(full["error_m"] - spec["paper_autots_error_m"]) > 0.05:
        notes.append(
            f"AutoTS error differs from paper {spec['paper_autots_error_m']:.2f} m"
        )
    if abs(alt["error_m"] - spec["paper_ablation_error_m"]) > 0.05:
        notes.append(
            f"ablation error differs from paper {spec['paper_ablation_error_m']:.2f} m"
        )
    if len(sample.get("location_point", [])) <= 1:
        notes.append("only one cached location point")
    return "; ".join(notes) if notes else "matches paper reference values"


def paste_fit(canvas, img, box):
    x1, y1, x2, y2 = box
    img = img.copy()
    img.thumbnail((x2 - x1, y2 - y1))
    canvas.paste(img, (x1 + (x2 - x1 - img.width) // 2,
                       y1 + (y2 - y1 - img.height) // 2))


def draw_header(draw, columns):
    for label, x1, x2 in columns:
        draw.text((x1 + 4, 18), label, fill="black")
        draw.line([x1, 0, x1, 40], fill="#555555")
        draw.line([x2, 0, x2, 40], fill="#555555")
    draw.line([0, 40, columns[-1][2], 40], fill="#555555", width=1)


def load_raw_sign_lookup(cfg):
    lookup = {}
    for split, path in [("train", cfg.DATA.TRAIN_JSON), ("test", cfg.DATA.TEST_JSON)]:
        with open(path) as f:
            data = json.load(f)
        for sign_id, sign_data in data.items():
            lookup[(split, str(sign_id))] = sign_data
    return lookup


def web_image_path(img_root, frame_id):
    root = img_root.replace("\\", "/")
    if root.startswith("./"):
        root = root[2:]
    return f"{root}/{frame_id}.png"


def to_float_list(values):
    if values is None:
        return None
    return [float(x) for x in values]


def box_to_web(box):
    if box is None:
        return None
    return [float(x) for x in box]


def image_size(path):
    with Image.open(path) as img:
        return list(img.size)


def build_web_frames(sample, raw_sign_data, img_root):
    detected_points = {
        frame_id: to_float_list(point)
        for frame_id, point in zip(
            sample.get("point_frames", []),
            sample.get("location_point", []),
        )
    }
    gt_points = {
        frame_id: to_float_list(point)
        for frame_id, point in zip(
            sample.get("gt_point_frames", []),
            sample.get("gt_location_point", []),
        )
    }
    image_geolocations = raw_sign_data.get("image_geolocations", {}) if raw_sign_data else {}
    frames = []
    for frame_id in sample.get("frame_ids", list(sample.get("gt_boxes", {}).keys())):
        path = os.path.join(img_root, f"{frame_id}.png")
        frames.append({
            "frame_id": frame_id,
            "image_src": web_image_path(img_root, frame_id),
            "image_size": image_size(path),
            "camera_gps": to_float_list(image_geolocations.get(frame_id)),
            "gt_box": box_to_web(sample.get("gt_boxes", {}).get(frame_id)),
            "detected_box": box_to_web(sample.get("detected_boxes", {}).get(frame_id)),
            "detected_score": (
                float(sample.get("detected_scores", {}).get(frame_id))
                if frame_id in sample.get("detected_scores", {}) else None
            ),
            "detected_location_point": detected_points.get(frame_id),
            "gt_location_point": gt_points.get(frame_id),
        })
    return frames


def build_web_row(idx, sample, spec, reason, pred, full, alt, deviation,
                  img_root, raw_sign_data):
    point_frames = sample.get("point_frames", [])
    outliers = set(full.get("outlier_indices", []))
    location_points = []
    for point_idx, point in enumerate(sample.get("location_point", [])):
        frame_id = point_frames[point_idx] if point_idx < len(point_frames) else None
        location_points.append({
            "index": point_idx,
            "frame_id": frame_id,
            "lat": float(point[0]),
            "lon": float(point[1]),
            "is_outlier": point_idx in outliers,
        })

    gt_point_frames = sample.get("gt_point_frames", [])
    gt_location_points = []
    for point_idx, point in enumerate(sample.get("gt_location_point", [])):
        frame_id = gt_point_frames[point_idx] if point_idx < len(gt_point_frames) else None
        gt_location_points.append({
            "index": point_idx,
            "frame_id": frame_id,
            "lat": float(point[0]),
            "lon": float(point[1]),
        })

    return {
        "index": idx,
        "split": sample["split"],
        "sign_id": str(sample["sign_id"]),
        "selection_reason": reason,
        "matches_paper_sample": (
            sample["split"] == spec["split"]
            and str(sample["sign_id"]) == spec["sign_id"]
        ),
        "category": int(sample["category"]),
        "paper_label": spec["paper_sign_label"],
        "gt_direction": int(sample["direction"]),
        "gt_direction_name": ORIENTATION_NAMES.get(int(sample["direction"])),
        "pred_direction": int(pred),
        "pred_direction_name": ORIENTATION_NAMES.get(int(pred)),
        "ablation": spec["ablation"],
        "gt_location": to_float_list(sample["geolocation"]),
        "autots": {
            "pred": to_float_list(full["pred"]),
            "error_m": float(full["error_m"]),
            "outlier_indices": [int(x) for x in full.get("outlier_indices", [])],
        },
        "ablation_result": {
            "pred": to_float_list(alt["pred"]),
            "error_m": float(alt["error_m"]),
        },
        "paper_reference": {
            "sign_label": spec["paper_sign_label"],
            "gt_direction": int(spec["paper_gt_direction"]),
            "gt_direction_name": ORIENTATION_NAMES.get(spec["paper_gt_direction"]),
            "pred_direction": int(spec["paper_pred_direction"]),
            "pred_direction_name": ORIENTATION_NAMES.get(spec["paper_pred_direction"]),
            "autots_error_m": float(spec["paper_autots_error_m"]),
            "ablation_error_m": float(spec["paper_ablation_error_m"]),
            "caption_geolocation": to_float_list(spec["paper_caption_geolocation"]),
        },
        "paper_deviation": deviation,
        "location_points": location_points,
        "gt_location_points": gt_location_points,
        "frames": build_web_frames(sample, raw_sign_data, img_root),
    }


def write_location_points_csv(path, web_rows):
    ensure_dir(os.path.dirname(path))
    frame_lookup = {}
    for row in web_rows:
        frame_lookup[(row["split"], row["sign_id"])] = {
            frame["frame_id"]: frame for frame in row["frames"]
        }

    fields = [
        "Figure8_Index",
        "ParentID",
        "Split",
        "SignID",
        "Category",
        "PaperLabel",
        "PointRole",
        "PointIndex",
        "FrameID",
        "Latitude",
        "Longitude",
        "RenderedInEstimatedMap",
        "IsOutlier",
        "CameraLatitude",
        "CameraLongitude",
        "GTSignLatitude",
        "GTSignLongitude",
        "AutoTSPredLatitude",
        "AutoTSPredLongitude",
        "AutoTSError_m",
        "Ablation",
        "AblationPredLatitude",
        "AblationPredLongitude",
        "AblationError_m",
        "GTDirection",
        "PredDirection",
        "PaperCaptionLatitude",
        "PaperCaptionLongitude",
        "PaperAutoTSError_m",
        "PaperAblationError_m",
        "PaperGTDirection",
        "PaperPredDirection",
    ]

    def frame_coord(frame, key):
        point = frame.get(key) if frame else None
        if point:
            return point[0], point[1]
        return "", ""

    rows = []
    for row in web_rows:
        parent_id = f"{row['split']}/{row['sign_id']}"
        frames = frame_lookup[(row["split"], row["sign_id"])]

        def append_point(point, role, rendered, is_outlier):
            frame = frames.get(point.get("frame_id"))
            camera_lat, camera_lon = frame_coord(frame, "camera_gps")
            paper = row["paper_reference"]
            paper_lat, paper_lon = paper["caption_geolocation"]
            rows.append({
                "Figure8_Index": row["index"],
                "ParentID": parent_id,
                "Split": row["split"],
                "SignID": row["sign_id"],
                "Category": row["category"],
                "PaperLabel": row["paper_label"],
                "PointRole": role,
                "PointIndex": point["index"],
                "FrameID": point.get("frame_id") or "",
                "Latitude": f"{point['lat']:.12f}",
                "Longitude": f"{point['lon']:.12f}",
                "RenderedInEstimatedMap": rendered,
                "IsOutlier": is_outlier,
                "CameraLatitude": f"{camera_lat:.12f}" if camera_lat != "" else "",
                "CameraLongitude": f"{camera_lon:.12f}" if camera_lon != "" else "",
                "GTSignLatitude": f"{row['gt_location'][0]:.12f}",
                "GTSignLongitude": f"{row['gt_location'][1]:.12f}",
                "AutoTSPredLatitude": f"{row['autots']['pred'][0]:.12f}",
                "AutoTSPredLongitude": f"{row['autots']['pred'][1]:.12f}",
                "AutoTSError_m": f"{row['autots']['error_m']:.2f}",
                "Ablation": row["ablation"],
                "AblationPredLatitude": f"{row['ablation_result']['pred'][0]:.12f}",
                "AblationPredLongitude": f"{row['ablation_result']['pred'][1]:.12f}",
                "AblationError_m": f"{row['ablation_result']['error_m']:.2f}",
                "GTDirection": row["gt_direction_name"],
                "PredDirection": row["pred_direction_name"],
                "PaperCaptionLatitude": f"{paper_lat:.12f}",
                "PaperCaptionLongitude": f"{paper_lon:.12f}",
                "PaperAutoTSError_m": f"{paper['autots_error_m']:.2f}",
                "PaperAblationError_m": f"{paper['ablation_error_m']:.2f}",
                "PaperGTDirection": paper["gt_direction_name"],
                "PaperPredDirection": paper["pred_direction_name"],
            })

        for point in row["location_points"]:
            append_point(
                point,
                "detected_location_point",
                "yes",
                "yes" if point.get("is_outlier") else "no",
            )
        for point in row["gt_location_points"]:
            append_point(point, "gt_box_location_point", "no", "")

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate qualitative Figure 8 reproduction.")
    parser.add_argument("--cfg", default="./orientation_config_v2_paper.yaml")
    parser.add_argument("--cache-dir", default="./results/orientation_cache")
    parser.add_argument("--orientation-ckpt",
                        default="./results/table4_orientation/AutoTS_ours_best.pth")
    parser.add_argument("--out", default="./results/figure8/figure8_qualitative.png")
    parser.add_argument("--web-data-out",
                        default="./results/figure8/figure8_web_data.json")
    parser.add_argument("--points-csv-out",
                        default="./results/figure8/figure8_location_points.csv")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    samples, all_samples = load_samples(args.cache_dir, args.limit)
    raw_sign_lookup = load_raw_sign_lookup(cfg)

    model = OrientationAblationNet(cfg, encoder="transformer").to(cfg.MODEL.DEVICE)
    state = safe_torch_load(args.orientation_ckpt, map_location=cfg.MODEL.DEVICE)
    model.load_state_dict(state)
    model.eval()

    row_samples = []
    used_keys = set()
    for spec in PAPER_FIG8_ROWS:
        sample, reason = select_paper_sample(spec, samples, all_samples, used_keys)
        row_samples.append((sample, spec, reason))

    init_projection_transforms()
    ensure_dir(os.path.dirname(args.out))
    width, row_h = 1500, 240
    header_h = 45
    height = header_h + row_h * len(row_samples) + 70
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    columns = [
        ("Index", 0, 55),
        ("Image Sequence", 55, 470),
        ("GT GPS Location", 470, 705),
        ("Estimated Location and Absolute Error", 705, 1210),
        ("GT and Estimated Orientation", 1210, width),
    ]
    draw_header(draw, columns)

    roots = {
        "train": cfg.DATA.TRAIN_IMG_ROOT,
        "test": cfg.DATA.TEST_IMG_ROOT,
    }
    csv_rows = []
    web_rows = []
    metadata = {
        "output": args.out,
        "web_data_output": args.web_data_out,
        "cache_dir": args.cache_dir,
        "orientation_ckpt": args.orientation_ckpt,
        "map_rendering": {
            "gt_gps_location": "Google satellite tiles, lyrs=s",
            "estimated_location": "Google road map tiles, lyrs=m",
            "preferred_zoom": 20,
            "fallback_zoom": 18,
            "gt_min_side_m": 80.0,
            "estimated_min_side_m": 120.0,
        },
        "paper_reference_note": (
            "Rendered predictions and errors are recomputed from the current "
            "cache/checkpoint. Paper values are stored for comparison only."
        ),
        "rows": [],
    }

    for idx, (sample, spec, reason) in enumerate(row_samples, start=1):
        ablation = spec["ablation"]
        y0 = header_h + (idx - 1) * row_h
        draw.line([0, y0, width, y0], fill="#777777")
        draw.text((22, y0 + row_h // 2 - 5), str(idx), fill="black")
        root = roots[sample["split"]]
        seq_img = draw_boxed_sequence(sample, root, "detected_boxes",
                                      max_images=2, size=(395, 205),
                                      label=spec["paper_sign_label"])
        paste_fit(canvas, seq_img, (65, y0 + 12, 460, y0 + row_h - 12))

        gt_panel = render_points_panel(
            sample["location_point"], sample["geolocation"],
            [("GT", sample["geolocation"])], "GT GPS Location",
            tile_url_template=GOOGLE_SATELLITE_TILE_URL,
            preferred_zoom=20,
            fallback_zoom=18,
            min_side_m=80.0,
        )
        paste_fit(canvas, gt_panel, (480, y0 + 15, 695, y0 + row_h - 15))

        full = localization_result(sample["location_point"], sample["geolocation"], "full")
        if ablation == "AutoTS w/o MinCut":
            alt = localization_result(sample["location_point"], sample["geolocation"], "no_mincut")
            alt_mode = "no_mincut"
        else:
            alt = localization_result(sample["location_point"], sample["geolocation"], "no_weight")
            alt_mode = "no_weight"
        full_panel = render_points_panel(
            sample["location_point"], sample["geolocation"],
            [("AutoTS", full["pred"])],
            f"AutoTS\nError: {full['error_m']:.2f} m",
            full["outlier_indices"],
            tile_url_template=GOOGLE_ROAD_TILE_URL,
            preferred_zoom=20,
            fallback_zoom=18,
            min_side_m=120.0,
        )
        alt_panel = render_points_panel(
            sample["location_point"], sample["geolocation"],
            [(ablation, alt["pred"])],
            f"{ablation}\nError: {alt['error_m']:.2f} m",
            [] if alt_mode == "no_mincut" else full["outlier_indices"],
            tile_url_template=GOOGLE_ROAD_TILE_URL,
            preferred_zoom=20,
            fallback_zoom=18,
            min_side_m=120.0,
        )
        paste_fit(canvas, full_panel, (715, y0 + 15, 955, y0 + row_h - 15))
        paste_fit(canvas, alt_panel, (965, y0 + 15, 1200, y0 + row_h - 15))

        pred = predict_sample(cfg, model, sample)
        orient_img = orientation_panel(int(sample["direction"]), pred, "AutoTS")
        paste_fit(canvas, orient_img, (1220, y0 + 40, width - 20, y0 + row_h - 40))
        deviation = build_paper_deviation(spec, sample, pred, full, alt)

        csv_rows.append([
            idx, sample["split"], sample["sign_id"], sample["category"],
            sample["direction"], pred, ablation,
            f"{full['error_m']:.2f}", f"{alt['error_m']:.2f}",
        ])
        metadata["rows"].append({
            "index": idx,
            "split": sample["split"],
            "sign_id": sample["sign_id"],
            "selection_reason": reason,
            "category": sample["category"],
            "gt_direction": sample["direction"],
            "pred_direction": pred,
            "ablation": ablation,
            "paper_reference": {
                "sign_label": spec["paper_sign_label"],
                "gt_direction": spec["paper_gt_direction"],
                "pred_direction": spec["paper_pred_direction"],
                "autots_error_m": spec["paper_autots_error_m"],
                "ablation_error_m": spec["paper_ablation_error_m"],
                "caption_geolocation": spec["paper_caption_geolocation"],
            },
            "matches_paper_sample": (
                sample["split"] == spec["split"]
                and str(sample["sign_id"]) == spec["sign_id"]
            ),
            "paper_deviation": deviation,
            "autots_error_m": full["error_m"],
            "ablation_error_m": alt["error_m"],
            "num_location_points": len(sample["location_point"]),
        })
        web_rows.append(build_web_row(
            idx, sample, spec, reason, pred, full, alt, deviation,
            root, raw_sign_lookup.get((sample["split"], str(sample["sign_id"]))),
        ))

    draw.line([0, header_h + row_h * len(row_samples), width,
               header_h + row_h * len(row_samples)], fill="#777777")
    caption = (
        "Fig. 8 reproduction: GT=satellite, estimates=road map; "
        "paper refs in metadata."
    )
    draw.text((40, height - 50), caption, fill="black")
    canvas.save(args.out)

    csv_path = os.path.splitext(args.out)[0] + "_errors.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Index", "Split", "SignID", "Category", "GT_Direction",
            "Pred_Direction", "Ablation", "AutoTS_Error_m", "Ablation_Error_m",
        ])
        writer.writerows(csv_rows)

    with open(os.path.splitext(args.out)[0] + "_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    web_payload = {
        "source": "AutoTS Figure 8 reproduction web data",
        "cache_dir": args.cache_dir,
        "orientation_ckpt": args.orientation_ckpt,
        "static_png": args.out,
        "errors_csv": csv_path,
        "metadata_json": os.path.splitext(args.out)[0] + "_metadata.json",
        "location_points_csv": args.points_csv_out,
        "map_tiles": {
            "gt_satellite": GOOGLE_SATELLITE_TILE_URL,
            "estimated_road": GOOGLE_ROAD_TILE_URL,
            "default_zoom": 18,
            "max_zoom": 20,
        },
        "note": (
            "Rendered values are recomputed from the current cache/checkpoint. "
            "Paper values are references only."
        ),
        "rows": web_rows,
    }
    ensure_dir(os.path.dirname(args.web_data_out))
    with open(args.web_data_out, "w") as f:
        json.dump(web_payload, f, indent=2)
    num_point_rows = write_location_points_csv(args.points_csv_out, web_rows)
    print(f"Saved Figure 8 to {args.out}")
    print(f"Saved Figure 8 web data to {args.web_data_out}")
    print(f"Saved {num_point_rows} Figure 8 point rows to {args.points_csv_out}")


if __name__ == "__main__":
    main()
