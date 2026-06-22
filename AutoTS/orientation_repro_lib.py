import csv
import json
import os
import random
import time
from collections import OrderedDict
import io
import urllib.request

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from geopy import Point as GeoPoint
from geopy import distance as geo_distance
from geopy.distance import geodesic
from scipy.spatial.distance import cdist
from shapely.geometry import Point
from shapely.ops import transform as shapely_transform
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import Dataset
from torchvision import models, transforms
from tqdm import tqdm
from yacs.config import CfgNode as CN

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


ORIENTATION_NAMES = {0: "Leftward", 1: "Backward", 2: "Rightward"}
PAPER_TABLE_IV = [
    ("AutoTS w/ ROI", 71.43, 12.50, 91.30, 22.22, 42.01),
    ("AutoTS w/ ROIS", 73.02, 37.50, 84.78, 44.44, 55.57),
    ("AutoTS w/o SIFT", 76.19, 37.50, 89.13, 44.44, 62.38),
    ("AutoTS w/o MImg", 80.95, 50.00, 89.13, 66.67, 68.60),
    ("AutoTS w/ BiLSTMs", 82.54, 62.50, 91.30, 55.56, 69.79),
    ("AutoTS w/ LSTMs", 82.54, 50.00, 93.48, 55.56, 67.07),
    ("AutoTS (ours)", 85.71, 50.00, 95.65, 66.67, 70.77),
    ("AutoTS\u2020", 88.89, 75.00, 95.65, 66.67, 79.11),
]

CENTER_X = 621
FOV_X = 45
OFFSET_X = 2
OFFSET_Y = 4
_TO_LOCAL = None
_TO_GEO = None


def load_cfg(path):
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(path)
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def split_cache_path(cache_dir, split, limit=None):
    suffix = f"_limit{limit}" if limit is not None else ""
    return os.path.join(cache_dir, f"{split}_features{suffix}.pt")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    ix1, iy1 = max(x1, x3), max(y1, y3)
    ix2, iy2 = min(x2, x4), min(y2, y4)
    inter = max(0, ix2 - ix1 + 1) * max(0, iy2 - iy1 + 1)
    area1 = (x2 - x1 + 1) * (y2 - y1 + 1)
    area2 = (x4 - x3 + 1) * (y4 - y3 + 1)
    denom = float(area1 + area2 - inter)
    return inter / denom if denom else 0.0


def build_detector(config_path, weight_path, device):
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weight_path
    cfg.MODEL.DEVICE = device
    return DefaultPredictor(cfg)


def build_resnet_extractor(device):
    try:
        weights = models.ResNet101_Weights.IMAGENET1K_V1
        resnet_model = models.resnet101(weights=weights)
    except AttributeError:
        resnet_model = models.resnet101(pretrained=True)
    resnet_model = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))
    resnet_model = resnet_model.to(device)
    resnet_model.eval()
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return resnet_model, preprocess


def extract_image_features(images, extractor, preprocess, device):
    tensors = [preprocess(Image.fromarray(img)) for img in images]
    batch = torch.stack(tensors, dim=0).to(device)
    with torch.no_grad():
        features = extractor(batch).view(len(images), -1)
    return features.detach().cpu()


def select_detected_sequence(frame_ids, gt_boxes, category, instances_by_frame):
    selected_features = []
    selected_boxes = OrderedDict()
    selected_scores = OrderedDict()
    for frame_id in frame_ids:
        info = instances_by_frame[frame_id]
        boxes = info["boxes"]
        classes = info["classes"]
        scores = info["scores"]
        roi_features = info["roi_features"]
        # Associate each frame's detection to that frame's GT box independently.
        # The previous greedy frame-to-frame chain (iou(prev_detected_box, box))
        # broke on small/edge-drifting KITTI signs whose consecutive boxes have
        # zero overlap, collapsing the track to a single point. Matching against
        # the per-frame GT box recovers every frame the detector fires on.
        chosen = None
        best_iou = 0.5
        for idx, box in enumerate(boxes):
            if category != classes[idx]:
                continue
            overlap = iou(gt_boxes[frame_id], box)
            if overlap > best_iou:
                best_iou = overlap
                chosen = idx
        if chosen is None:
            continue
        selected_features.append(torch.as_tensor(roi_features[chosen], dtype=torch.float32))
        selected_boxes[frame_id] = [float(x) for x in boxes[chosen]]
        selected_scores[frame_id] = float(scores[chosen])
    if not selected_features:
        selected_features = [torch.zeros(1024, dtype=torch.float32)]
    return selected_features, selected_boxes, selected_scores


def extract_sift_descriptors(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    if descriptors is None:
        return None
    h, w = image.shape[:2]
    descriptors_loc = []
    mean_value = np.mean(np.nonzero(descriptors))
    for kp, desc in zip(keypoints, descriptors):
        x, y = kp.pt
        x_norm = x / max(w, 1) * mean_value
        y_norm = y / max(h, 1) * mean_value
        pos_encoding = np.array([x_norm] * 5 + [y_norm] * 5)
        full_descriptor = np.concatenate((pos_encoding, desc))
        descriptors_loc.append(full_descriptor)
    descriptors_loc = np.array(descriptors_loc, dtype=np.float32)
    norms = np.linalg.norm(descriptors_loc, axis=1, keepdims=True) + 1e-8
    return descriptors_loc / norms


def sift_to_vec(descriptor_groups, output_dim=160):
    descriptors = np.vstack(descriptor_groups)
    if len(descriptors) <= 1:
        return np.zeros((1, output_dim), dtype=np.float32)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1.6,
        linkage="ward",
    )
    clustering.fit(descriptors)
    labels = clustering.labels_
    img_features = np.zeros((len(descriptor_groups), output_dim), "float32")
    count = 0
    for i, group in enumerate(descriptor_groups):
        for j in range(len(group)):
            label = labels[j + count]
            if label < output_dim:
                img_features[i][label] += 1
        count += len(group)
    return img_features


def build_sift_features(images_by_frame, boxes_by_frame, device="cpu"):
    if not boxes_by_frame:
        return [torch.zeros(160, dtype=torch.float32)], []
    descriptor_groups = []
    used_frames = []
    last_box = list(boxes_by_frame.values())[-1]
    last_h = max(int(last_box[3] - last_box[1]), 1)
    for frame_id, box in boxes_by_frame.items():
        if frame_id not in images_by_frame:
            continue
        xmin, ymin, xmax, ymax = map(int, box)
        image = images_by_frame[frame_id]
        h, w = image.shape[:2]
        xmin, xmax = max(0, xmin), min(w, xmax)
        ymin, ymax = max(0, ymin), min(h, ymax)
        if xmax <= xmin or ymax <= ymin:
            continue
        roi = image[ymin:ymax, xmin:xmax]
        cur_h = max(ymax - ymin, 1)
        scale = last_h / cur_h
        new_w = max(int(scale * (xmax - xmin)), 1)
        new_h = max(int(scale * cur_h), 1)
        roi_resized = cv2.resize(roi, (new_w, new_h))
        descriptors = extract_sift_descriptors(roi_resized)
        if descriptors is not None:
            descriptor_groups.append(descriptors)
            used_frames.append(frame_id)
    if not descriptor_groups:
        return [torch.zeros(160, dtype=torch.float32)], []
    sift_arr = sift_to_vec(descriptor_groups)
    return [torch.tensor(vec, dtype=torch.float32) for vec in sift_arr], used_frames


def get_geo(lat, lng, depth, alt):
    if 0 <= alt <= 90:
        alt = 90 - alt
    if alt > 90:
        alt = 450 - alt
    if alt < 0:
        alt = 90 - alt
    start = GeoPoint(latitude=lat, longitude=lng)
    dest = geo_distance.distance(meters=float(depth)).destination(start, alt)
    return dest.latitude, dest.longitude


def compute_location_points(sign_data, boxes_by_frame, depth_dict):
    image_yaws = sign_data.get("image_yaws", {})
    geolocs = sign_data.get("image_geolocations", {})
    points = []
    point_frames = []
    for frame_id, box in boxes_by_frame.items():
        depth_crop = depth_dict.get(frame_id)
        if depth_crop is None:
            continue
        xmin = int(box[0]) + OFFSET_X
        ymin = int(box[1]) + OFFSET_Y
        xmax = int(box[2]) - OFFSET_X
        ymax = int(box[3]) - OFFSET_Y
        if xmax <= xmin or ymax <= ymin:
            continue
        depth_map = depth_crop[ymin:ymax, xmin:xmax]
        depth_values = depth_map[np.nonzero(depth_map)]
        if len(depth_values) == 0:
            continue
        depth = np.median(depth_values)
        center_x = (box[0] + box[2]) / 2
        relative_angle = (center_x - CENTER_X) * (FOV_X / CENTER_X)
        yaw = np.degrees(image_yaws.get(frame_id, 0))
        alt = yaw - relative_angle
        lat0, lng0 = geolocs.get(frame_id, (None, None))
        if lat0 is None or lng0 is None:
            continue
        points.append(list(get_geo(lat0, lng0, depth, alt)))
        point_frames.append(frame_id)
    return points, point_frames


def safe_torch_save(obj, path):
    ensure_dir(os.path.dirname(path))
    torch.save(obj, path)


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_or_load_cache(cfg, cache_dir, split, json_path, img_root, limit=None,
                        force=False):
    ensure_dir(cache_dir)
    path = split_cache_path(cache_dir, split, limit)
    if os.path.exists(path) and not force:
        payload = safe_torch_load(path, map_location="cpu")
        if limit is None or payload.get("limit") == limit:
            return payload

    print(f"Building {split} orientation cache...")
    t0 = time.time()
    data = load_json(json_path)
    keys = list(data.keys())
    if limit is not None:
        keys = keys[:limit]

    detector = build_detector(cfg.DETECTRON2.CONFIG_PATH,
                              cfg.DETECTRON2.WEIGHT_PATH,
                              cfg.MODEL.DEVICE)
    img_extractor, preprocess = build_resnet_extractor(cfg.MODEL.DEVICE)
    depth_npz = np.load("./data/img_depth.npz")
    depth_dict = {k: depth_npz[k] for k in depth_npz.files}

    samples = []
    for sign_id in tqdm(keys, desc=f"Caching {split}"):
        sign_data = data[sign_id]
        frame_ids = list(sign_data["images"].keys())
        gt_boxes = OrderedDict(
            (frame_id, [float(x) for x in sign_data["images"][frame_id]])
            for frame_id in frame_ids
        )
        images_by_frame = OrderedDict()
        for frame_id in frame_ids:
            img_path = os.path.join(img_root, f"{frame_id}.png")
            images_by_frame[frame_id] = np.array(Image.open(img_path).convert("RGB"))

        images = list(images_by_frame.values())
        image_feature = extract_image_features(
            images, img_extractor, preprocess, cfg.MODEL.DEVICE
        )

        instances_by_frame = OrderedDict()
        for frame_id, image in images_by_frame.items():
            with torch.no_grad():
                outputs, roi_features = detector(image)
            instances = outputs[0]["instances"].to("cpu")
            boxes = instances.pred_boxes.tensor.numpy().tolist()
            classes = instances.pred_classes.numpy().tolist()
            scores = instances.scores.numpy().tolist()
            instances_by_frame[frame_id] = {
                "boxes": boxes,
                "classes": classes,
                "scores": scores,
                "roi_features": roi_features,
            }

        category = int(sign_data["category"])
        roi_detected, detected_boxes, detected_scores = select_detected_sequence(
            frame_ids, gt_boxes, category, instances_by_frame
        )
        gt_roi_features, _, _ = select_detected_sequence(
            frame_ids, gt_boxes, category, instances_by_frame
        )
        if not gt_roi_features:
            gt_roi_features = [torch.zeros(1024, dtype=torch.float32)]

        sift_detected, sift_detected_frames = build_sift_features(
            images_by_frame, detected_boxes
        )
        sift_gt, sift_gt_frames = build_sift_features(images_by_frame, gt_boxes)
        detected_points, detected_point_frames = compute_location_points(
            sign_data, detected_boxes, depth_dict
        )
        gt_points, gt_point_frames = compute_location_points(
            sign_data, gt_boxes, depth_dict
        )

        samples.append({
            "split": split,
            "sign_id": str(sign_id),
            "category": category,
            "direction": int(sign_data["direction"]),
            "geolocation": [float(x) for x in sign_data["Geolocation"]],
            "frame_ids": frame_ids,
            "image_feature": image_feature.cpu(),
            "roi_feature": [x.detach().cpu() for x in roi_detected],
            "gt_roi_feature": [x.detach().cpu() for x in gt_roi_features],
            "sift_feature": [x.detach().cpu() for x in sift_detected],
            "gt_sift_feature": [x.detach().cpu() for x in sift_gt],
            "gt_boxes": dict(gt_boxes),
            "detected_boxes": dict(detected_boxes),
            "detected_scores": dict(detected_scores),
            "location_point": detected_points,
            "gt_location_point": gt_points,
            "point_frames": detected_point_frames,
            "gt_point_frames": gt_point_frames,
            "sift_frames": sift_detected_frames,
            "gt_sift_frames": sift_gt_frames,
        })

    payload = {
        "split": split,
        "limit": limit,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples": samples,
        "notes": [
            "GT-box variant uses GT boxes for SIFT and location points.",
            "ROI features for GT-box variant use matched detector ROI features because the bundled DefaultPredictor exposes ROI features only for predicted boxes.",
        ],
    }
    safe_torch_save(payload, path)
    print(f"Saved {split} cache to {path} in {time.time() - t0:.1f}s")
    return payload


class CachedOrientationDataset(Dataset):
    def __init__(self, samples, variant, device):
        self.samples = samples
        self.variant = variant
        self.device = device

    def __len__(self):
        return len(self.samples)

    def _select_sequence(self, values, mode):
        if mode == "single":
            idx = len(values) // 2
            return [values[idx]]
        return list(values)

    def _zeros_like_image(self, sample):
        return torch.zeros_like(sample["image_feature"])

    def __getitem__(self, idx):
        sample = self.samples[idx]
        use_gt = self.variant == "AutoTS\u2020"

        if self.variant == "AutoTS w/ ROI":
            roi = self._select_sequence(sample["roi_feature"], "single")
            sift = [torch.zeros(160, dtype=torch.float32) for _ in roi]
            img = torch.zeros((1, 2048), dtype=torch.float32)
        elif self.variant == "AutoTS w/ ROIS":
            roi = sample["roi_feature"]
            sift = [torch.zeros(160, dtype=torch.float32) for _ in roi]
            img = torch.zeros_like(sample["image_feature"])
        else:
            roi = sample["gt_roi_feature"] if use_gt else sample["roi_feature"]
            if self.variant == "AutoTS w/o SIFT":
                sift = [torch.zeros(160, dtype=torch.float32) for _ in roi]
            else:
                sift = sample["gt_sift_feature"] if use_gt else sample["sift_feature"]
            img = sample["image_feature"]
            if self.variant == "AutoTS w/o MImg":
                img = torch.zeros_like(img)

        if len(sift) != len(roi):
            if len(sift) == 0:
                sift = [torch.zeros(160, dtype=torch.float32) for _ in roi]
            elif len(sift) < len(roi):
                sift = list(sift) + [sift[-1]] * (len(roi) - len(sift))
            else:
                sift = list(sift[:len(roi)])

        return {
            "image_feature": img.float(),
            "roi_feature": [x.float() for x in roi],
            "sift_feature": [x.float() for x in sift],
            "direction": int(sample["direction"]),
            "geolocation": sample["geolocation"],
            "location_point": sample["gt_location_point"] if use_gt else sample["location_point"],
            "category": int(sample["category"]),
            "sign_id": sample["sign_id"],
            "split": sample["split"],
        }


def orientation_collate(batch):
    return {
        "image_feature": [item["image_feature"] for item in batch],
        "roi_feature": [item["roi_feature"] for item in batch],
        "sift_feature": [item["sift_feature"] for item in batch],
        "direction": [item["direction"] for item in batch],
        "geolocation": [item["geolocation"] for item in batch],
        "location_point": [item["location_point"] for item in batch],
        "category": [item["category"] for item in batch],
        "sign_id": [item["sign_id"] for item in batch],
        "split": [item["split"] for item in batch],
    }


class OrientationAblationNet(nn.Module):
    def __init__(self, cfg, encoder="transformer"):
        super().__init__()
        self.device = cfg.MODEL.DEVICE
        self.encoder_type = encoder
        self.roi_fc = nn.Linear(cfg.MODEL.ROI_INPUT_SIZE, cfg.MODEL.ROI_HIDDEN)
        self.sift_fc = nn.Linear(cfg.MODEL.SIFT_INPUT_SIZE, cfg.MODEL.SIFT_HIDDEN)
        self.image_fc = nn.Linear(2048, cfg.MODEL.IMG_HIDDEN)
        d_model = cfg.MODEL.D_MODEL
        if encoder == "transformer":
            self.sequence_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=cfg.MODEL.NHEAD,
                    batch_first=True,
                ),
                num_layers=cfg.MODEL.NUM_LAYERS,
            )
            seq_dim = d_model
        elif encoder == "bilstm":
            hidden = d_model // 2
            self.sequence_encoder = nn.LSTM(
                d_model, hidden, num_layers=cfg.MODEL.NUM_LAYERS,
                batch_first=True, bidirectional=True
            )
            seq_dim = hidden * 2
        elif encoder == "lstm":
            self.sequence_encoder = nn.LSTM(
                d_model, d_model, num_layers=cfg.MODEL.NUM_LAYERS,
                batch_first=True
            )
            seq_dim = d_model
        else:
            raise ValueError(f"Unknown encoder: {encoder}")
        self.out_fc = nn.Linear(seq_dim + cfg.MODEL.IMG_HIDDEN, cfg.MODEL.NUM_CLASSES)

    def pad_tensor_list_batch(self, features, max_seq_len, pad_value=0.0):
        batch_size = len(features)
        feature_dim = features[0][0].shape[0]
        padded = torch.full(
            (batch_size, max_seq_len, feature_dim),
            pad_value,
            dtype=torch.float32,
            device=self.device,
        )
        for i, sample in enumerate(features):
            for j, tensor in enumerate(sample[:max_seq_len]):
                padded[i, j] = tensor.to(self.device)
        return padded

    def forward(self, data):
        img_input = data["image_feature"]
        roi_input = data["roi_feature"]
        sift_input = data["sift_feature"]
        max_seq_len = max(
            max(len(sample) for sample in img_input),
            max(len(sample) for sample in roi_input),
            max(len(sample) for sample in sift_input),
        )
        roi_input = self.pad_tensor_list_batch(roi_input, max_seq_len)
        sift_input = self.pad_tensor_list_batch(sift_input, max_seq_len)
        roi_emb = self.roi_fc(roi_input)
        sift_emb = self.sift_fc(sift_input)
        seq = torch.cat((roi_emb, sift_emb), dim=2)
        if self.encoder_type == "transformer":
            seq = self.sequence_encoder(seq)
        else:
            seq, _ = self.sequence_encoder(seq)
        seq_feature = seq.mean(dim=1)

        img_features = []
        for img in img_input:
            img = img.to(self.device)
            img_features.append(self.image_fc(img).mean(dim=0))
        img_features = torch.stack(img_features, dim=0)
        return self.out_fc(torch.cat((seq_feature, img_features), dim=1))


def class_weights(class_num, device):
    class_num = np.array(class_num, dtype=np.float32)
    weights = (1.0 / class_num) / np.sum(1.0 / class_num) * len(class_num)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_metrics(labels, preds, num_classes=3):
    labels = np.array(labels)
    preds = np.array(preds)
    acc = float(np.mean(preds == labels) * 100) if len(labels) else 0.0
    recalls = []
    for cls in range(num_classes):
        mask = labels == cls
        if np.sum(mask) == 0:
            recalls.append(0.0)
        else:
            recalls.append(float(np.mean(preds[mask] == labels[mask]) * 100))
    return {
        "accuracy": acc,
        "recall_left": recalls[0],
        "recall_back": recalls[1],
        "recall_right": recalls[2],
        "mrecall": float(np.mean(recalls)),
    }


def point_transform(location_point):
    if _TO_LOCAL is None:
        init_projection_transforms()
    return shapely_transform(_TO_LOCAL, Point(location_point))


def point_transform_back(local_point):
    if _TO_GEO is None:
        init_projection_transforms()
    return shapely_transform(_TO_GEO, Point(local_point))


def init_projection_transforms():
    global _TO_LOCAL, _TO_GEO
    import pyproj
    _TO_LOCAL = pyproj.Transformer.from_crs(
        "epsg:4326", "epsg:3044", always_xy=True
    ).transform
    _TO_GEO = pyproj.Transformer.from_crs(
        "epsg:3044", "epsg:4326", always_xy=True
    ).transform


def gaussian_affinity(coords, sigma=2.5):
    sq_dists = cdist(coords, coords, metric="sqeuclidean")
    return np.exp(-sq_dists / (2 * sigma ** 2))


def cluster_nsal(points, mode="full"):
    if not points:
        return None, []
    transformed = [point_transform(p) for p in points]
    coords = np.array([[p.x, p.y] for p in transformed])
    if len(coords) <= 1:
        return coords[0], []
    W = gaussian_affinity(coords)
    weight_sum = W.sum(axis=1)
    is_outlier = np.zeros(len(coords), dtype=bool)
    if mode in ("full", "no_weight"):
        sorted_weights = np.sort(weight_sum)
        diffs = np.diff(sorted_weights)
        mean_diff = np.mean(diffs) if len(diffs) > 0 else 0
        count = 0
        for diff in diffs:
            if diff > 3 * mean_diff:
                count += 1
            else:
                break
        num_del = min(int(len(coords) * 0.3), count)
        ws = weight_sum.copy()
        for _ in range(num_del):
            idx = np.argmin(ws)
            is_outlier[idx] = True
            ws[idx] = np.inf
    selected = coords[~is_outlier]
    if len(selected) == 0:
        return np.mean(coords, axis=0), np.where(is_outlier)[0].tolist()
    if mode == "no_weight":
        return np.mean(selected, axis=0), np.where(is_outlier)[0].tolist()
    weights = W.sum(axis=1)[~is_outlier]
    weight_total = np.sum(weights)
    if weight_total == 0:
        return np.mean(selected, axis=0), np.where(is_outlier)[0].tolist()
    norm_weights = weights / weight_total * len(selected)
    center = np.mean(selected * norm_weights[:, np.newaxis], axis=0)
    return center, np.where(is_outlier)[0].tolist()


def local_center_to_geo(center):
    pred = point_transform_back(center)
    return [pred.x, pred.y]


def localization_result(points, true_loc, mode="full"):
    center, outliers = cluster_nsal(points, mode=mode)
    if center is None:
        return None
    pred = local_center_to_geo(center)
    return {
        "pred": pred,
        "error_m": geodesic(pred, true_loc).meters,
        "outlier_indices": outliers,
    }


def draw_boxed_sequence(sample, img_root, boxes_key, max_images=2,
                        size=(420, 190), label=None):
    frame_ids = list(sample["gt_boxes"].keys())
    if len(frame_ids) > max_images:
        idxs = np.linspace(0, len(frame_ids) - 1, max_images).astype(int).tolist()
        frame_ids = [frame_ids[i] for i in idxs]
    canvas = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas)
    boxes = sample.get(boxes_key, {}) or sample.get("gt_boxes", {})
    y = 0
    for i, frame_id in enumerate(frame_ids):
        img_path = os.path.join(img_root, f"{frame_id}.png")
        img = Image.open(img_path).convert("RGB")
        img.thumbnail((size[0] - 30, size[1] // max_images + 30))
        x = 15 + i * 22
        y = i * (size[1] // max_images - 5)
        canvas.paste(img, (x, y))
        if frame_id in boxes:
            scale_x = img.width / Image.open(img_path).size[0]
            scale_y = img.height / Image.open(img_path).size[1]
            box = boxes[frame_id]
            rect = [x + box[0] * scale_x, y + box[1] * scale_y,
                    x + box[2] * scale_x, y + box[3] * scale_y]
            draw.rectangle(rect, outline="red", width=2)
            if label:
                try:
                    bbox = draw.textbbox((0, 0), label)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                except AttributeError:
                    text_w, text_h = draw.textsize(label)
                tx = max(x, min(rect[0], size[0] - text_w - 8))
                ty = max(y, rect[1] - text_h - 8)
                if ty < y + 2:
                    ty = min(rect[3] + 4, size[1] - text_h - 6)
                draw.rectangle([tx, ty, tx + text_w + 8, ty + text_h + 6],
                               fill="#ff3b1f")
                draw.text((tx + 4, ty + 3), label, fill="white")
    return canvas


TILE_CACHE = {}
GOOGLE_SATELLITE_TILE_URL = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
GOOGLE_ROAD_TILE_URL = "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}"

def download_tile(z, y, x, tile_url_template=GOOGLE_SATELLITE_TILE_URL):
    key = (tile_url_template, z, y, x)
    if key in TILE_CACHE:
        return TILE_CACHE[key]
    url = tile_url_template.format(x=x, y=y, z=z)
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=5) as response:
        img = Image.open(io.BytesIO(response.read()))
        TILE_CACHE[key] = img
        return img

def latlon_to_tile(lat, lon, zoom):
    lat_rad = np.radians(lat)
    n = 2.0 ** zoom
    xtile = (lon + 180.0) / 360.0 * n
    ytile = (1.0 - np.log(np.tan(lat_rad) + (1.0 / np.cos(lat_rad))) / np.pi) / 2.0 * n
    return xtile, ytile


def render_points_panel(points, true_loc, variants, title, outlier_indices=None,
                        tile_url_template=GOOGLE_SATELLITE_TILE_URL,
                        preferred_zoom=20, fallback_zoom=18,
                        min_side_m=30.0):
    fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=150)
    init_projection_transforms()
    
    # Calculate bounding box of all coordinates
    all_x = []
    all_y = []
    if points:
        for p in points:
            xy = point_transform(p)
            all_x.append(xy.x)
            all_y.append(xy.y)
    true_xy = point_transform(true_loc)
    all_x.append(true_xy.x)
    all_y.append(true_xy.y)
    for label, pred in variants:
        pred_xy = point_transform(pred)
        all_x.append(pred_xy.x)
        all_y.append(pred_xy.y)
        
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    
    # Pad and make square
    dx = xmax - xmin
    dy = ymax - ymin
    margin = max(dx, dy, 15.0) * 0.3
    xmin -= margin
    xmax += margin
    ymin -= margin
    ymax += margin
    
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    side = max(xmax - xmin, ymax - ymin, min_side_m)
    xmin = cx - side / 2.0
    xmax = cx + side / 2.0
    ymin = cy - side / 2.0
    ymax = cy + side / 2.0
    
    has_tiles = False
    try:
        # Get lat/lon of corners
        corners_x = [xmin, xmin, xmax, xmax]
        corners_y = [ymin, ymax, ymin, ymax]
        corners_lon, corners_lat = _TO_GEO(corners_x, corners_y)
        min_lat, max_lat = min(corners_lat), max(corners_lat)
        min_lon, max_lon = min(corners_lon), max(corners_lon)
        
        # Prefer the paper-like high zoom; fall back to zoom 18 if the panel
        # would require too many tiles.
        zoom_candidates = [preferred_zoom]
        if fallback_zoom != preferred_zoom:
            zoom_candidates.append(fallback_zoom)
        zoom = zoom_candidates[-1]
        for candidate in zoom_candidates:
            x_min_t, y_min_t = latlon_to_tile(max_lat, min_lon, candidate)
            x_max_t, y_max_t = latlon_to_tile(min_lat, max_lon, candidate)
            tile_x_start = int(np.floor(x_min_t))
            tile_x_end = int(np.floor(x_max_t))
            tile_y_start = int(np.floor(y_min_t))
            tile_y_end = int(np.floor(y_max_t))
            num_cols = tile_x_end - tile_x_start + 1
            num_rows = tile_y_end - tile_y_start + 1
            if num_cols * num_rows <= 16:
                zoom = candidate
                break
            
        # Download and stitch
        stitched_img = np.zeros((num_rows * 256, num_cols * 256, 3), dtype=np.uint8)
        for r in range(num_rows):
            for c in range(num_cols):
                tx = tile_x_start + c
                ty = tile_y_start + r
                tile = download_tile(zoom, ty, tx, tile_url_template)
                stitched_img[r*256:(r+1)*256, c*256:(c+1)*256] = np.array(tile.convert("RGB"))
                
        # Generate destination grid for warping
        W, H = 512, 512
        cols_grid = np.linspace(xmin, xmax, W)
        rows_grid = np.linspace(ymax, ymin, H)
        xx, yy = np.meshgrid(cols_grid, rows_grid)
        
        lons_grid, lats_grid = _TO_GEO(xx, yy)
        
        lat_rad = np.radians(lats_grid)
        n = 2.0 ** zoom
        xtile_frac = (lons_grid + 180.0) / 360.0 * n
        ytile_frac = (1.0 - np.log(np.tan(lat_rad) + (1.0 / np.cos(lat_rad))) / np.pi) / 2.0 * n
        
        map_x = (xtile_frac - tile_x_start) * 256.0
        map_y = (ytile_frac - tile_y_start) * 256.0
        
        warped = cv2.remap(
            stitched_img,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        ax.imshow(warped, extent=[xmin, xmax, ymin, ymax])
        has_tiles = True
    except Exception as e:
        print(f"Warning: tile retrieval/warping failed: {e}")
        
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    
    # Plot the points
    if points:
        coords = np.array([[point_transform(p).x, point_transform(p).y] for p in points])
        outlier_indices = set(outlier_indices or [])
        keep = [i for i in range(len(coords)) if i not in outlier_indices]
        outs = [i for i in range(len(coords)) if i in outlier_indices]
        if keep:
            ax.scatter(coords[keep, 0], coords[keep, 1], s=22, color="#f28e2b", label="points")
        if outs:
            ax.scatter(coords[outs, 0], coords[outs, 1], s=22, color="#9aa0a6", label="discarded")
            
    ax.scatter([true_xy.x], [true_xy.y], marker="*", s=80, color="red", label="GT")
    for label, pred in variants:
        pred_xy = point_transform(pred)
        ax.scatter([pred_xy.x], [pred_xy.y], marker="x", s=50, label=label)
        
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    if has_tiles:
        ax.grid(False)
    else:
        ax.grid(True, alpha=0.25)
    ax.legend(fontsize=5, loc="best")
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return Image.fromarray(arr)



def orientation_panel(gt, pred, title):
    img = Image.new("RGB", (250, 145), "white")
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([5, 5, 245, 58], outline="#cfd4da", width=2)
    draw.rounded_rectangle([5, 68, 245, 121], outline="#cfd4da", width=2)
    draw.rectangle([8, 8, 60, 55], fill="#dbe7f5")
    draw.rectangle([8, 71, 60, 118], fill="#e3f2d9" if gt == pred else "#ffd7bd")
    draw.text((22, 24), "GT", fill="black")
    draw.text((10, 84), "Estimated", fill="black")
    draw.text((88, 24), ORIENTATION_NAMES.get(gt, str(gt)), fill="black")
    draw.text((88, 84), ORIENTATION_NAMES.get(pred, str(pred)), fill="black")
    draw.text((5, 128), title, fill="black")
    return img


def write_csv(path, rows, header):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
