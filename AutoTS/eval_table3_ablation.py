"""
eval_table3_ablation.py — Reproduce Paper Table III
====================================================
Ablation study of NSAL method on KITTI-TS.

All methods use PlaneDepth depth estimation (same location points).
Only the clustering/aggregation step differs:

  1. AutoTS w/o MinCut — skip outlier removal, keep weighted center
  2. AutoTS w/o Weight — do MinCut, then simple mean (no weighting)
  3. AutoTS-K-Means   — use K-Means clustering
  4. AutoTS-DBSCAN    — use DBSCAN clustering
  5. AutoTS (ours)    — full NSAL (MinCut + Weighted center)

  * variants = evaluated only on sparse signs (<10 location points)

Metrics: MAE(m), RMSE(m), R@1m(%), R@2m(%)
Results computed over ALL samples (train + test).
"""

import json
import numpy as np
import csv
import os
import time
from geopy.distance import geodesic
from geopy import Point as GeoPoint, distance as geo_distance
from shapely.ops import transform
from shapely.geometry import Point
import pyproj
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial.distance import cdist

# ============================================================
# Camera parameters (KITTI default)
# ============================================================
CENTER_X = 621
FOV_X = 45
OFFSET_X = 2
OFFSET_Y = 4

# Cache transformers
_to_local = pyproj.Transformer.from_crs("epsg:4326", "epsg:3044", always_xy=True).transform
_to_geo = pyproj.Transformer.from_crs("epsg:3044", "epsg:4326", always_xy=True).transform


def load_all_signs():
    with open('./KITTI-TS/train/sign_id_GT.json') as f:
        train = json.load(f)
    with open('./KITTI-TS/test/sign_id_GT.json') as f:
        test = json.load(f)
    return {**train, **test}


def point_transform(location_point):
    geom = Point(location_point)
    return transform(_to_local, geom)


def point_transform_back(geo_point):
    geom = Point(geo_point)
    return transform(_to_geo, geom)


def gaussian_affinity(coords, sigma=2.5):
    sq_dists = cdist(coords, coords, metric='sqeuclidean')
    return np.exp(-sq_dists / (2 * sigma ** 2))


# ============================================================
# Location Point Extraction (PlaneDepth)
# ============================================================

def get_geo(lat, lng, depth, alt):
    if alt <= 90 and alt >= 0:
        alt = 90 - alt
    if alt > 90:
        alt = 450 - alt
    if alt < 0:
        alt = 90 - alt
    start = GeoPoint(latitude=lat, longitude=lng)
    destination = geo_distance.distance(meters=depth).destination(start, alt)
    return destination.latitude, destination.longitude


def compute_location_points(sign_data, boxes, depth_dict):
    image_yaws = sign_data.get('image_yaws', {})
    geolocs = sign_data.get('image_geolocations', {})
    if not image_yaws or not boxes:
        return []

    pos_list = []
    for obj_id, box in boxes.items():
        depth_map = depth_dict.get(obj_id, None)
        if depth_map is None:
            continue
        xmin = int(box[0]) + OFFSET_X
        ymin = int(box[1]) + OFFSET_Y
        xmax = int(box[2]) - OFFSET_X
        ymax = int(box[3]) - OFFSET_Y
        crop = depth_map[ymin:ymax, xmin:xmax]
        vals = crop[np.nonzero(crop)]
        if len(vals) == 0:
            continue
        depth = np.median(vals)
        if depth <= 0:
            continue

        center_x = (box[0] + box[2]) / 2
        relative_angle = (center_x - CENTER_X) * (FOV_X / CENTER_X)
        yaw = np.degrees(image_yaws.get(obj_id, 0))
        alt = yaw - relative_angle
        lat0, lng0 = geolocs.get(obj_id, (None, None))
        if lat0 is None:
            continue
        lat_, lng_ = get_geo(lat0, lng0, depth, alt)
        pos_list.append([lat_, lng_])
    return pos_list


def to_projected(points_list):
    """Convert GPS points to projected coordinates."""
    transformed = [point_transform(p) for p in points_list]
    return np.array([[p.x, p.y] for p in transformed])


def from_projected(center):
    """Convert projected center back to GPS."""
    pred = point_transform_back(center)
    return [pred.x, pred.y]


# ============================================================
# Clustering Methods (5 variants)
# ============================================================

def cluster_nsal(coords):
    """Full NSAL: MinCut + Weighted center (AutoTS ours)."""
    if len(coords) <= 1:
        return np.mean(coords, axis=0)

    W = gaussian_affinity(coords)
    weight_sum = W.sum(axis=1)

    # --- MinCut: outlier removal ---
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
    is_outlier = np.zeros(len(coords), dtype=bool)
    ws = weight_sum.copy()
    for _ in range(num_del):
        idx = np.argmin(ws)
        is_outlier[idx] = True
        ws[idx] = np.inf

    # --- Weighted center ---
    data_select = coords[~is_outlier]
    if len(data_select) == 0:
        return np.mean(coords, axis=0)

    weights = W.sum(axis=1)[~is_outlier]
    wt = np.sum(weights)
    if wt == 0:
        return np.mean(data_select, axis=0)
    norm_w = weights / wt * len(data_select)
    center = np.mean(data_select * norm_w[:, np.newaxis], axis=0)
    return center


def cluster_no_mincut(coords):
    """AutoTS w/o MinCut: skip outlier removal, keep weighted center."""
    if len(coords) <= 1:
        return np.mean(coords, axis=0)

    W = gaussian_affinity(coords)
    weights = W.sum(axis=1)
    wt = np.sum(weights)
    if wt == 0:
        return np.mean(coords, axis=0)
    norm_w = weights / wt * len(coords)
    center = np.mean(coords * norm_w[:, np.newaxis], axis=0)
    return center


def cluster_no_weight(coords):
    """AutoTS w/o Weight: MinCut + simple mean (no weighting)."""
    if len(coords) <= 1:
        return np.mean(coords, axis=0)

    W = gaussian_affinity(coords)
    weight_sum = W.sum(axis=1)

    # MinCut
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
    is_outlier = np.zeros(len(coords), dtype=bool)
    ws = weight_sum.copy()
    for _ in range(num_del):
        idx = np.argmin(ws)
        is_outlier[idx] = True
        ws[idx] = np.inf

    data_select = coords[~is_outlier]
    if len(data_select) == 0:
        return np.mean(coords, axis=0)

    # Simple mean instead of weighted
    return np.mean(data_select, axis=0)


def cluster_kmeans(coords):
    """AutoTS-K-Means: use K-Means clustering."""
    if len(coords) <= 1:
        return np.mean(coords, axis=0)

    # K-Means with k=1 just gives the centroid
    # With k>1, find the largest cluster
    # Paper says "intrinsic cluster center calculation method"
    # which is essentially centroid (K-Means with k=1)
    km = KMeans(n_clusters=1, random_state=42, n_init=10).fit(coords)
    return km.cluster_centers_[0]


def cluster_dbscan(coords, eps=5.0, min_samples=2):
    """AutoTS-DBSCAN: use DBSCAN clustering."""
    if len(coords) <= 2:
        return np.mean(coords, axis=0)

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = clustering.labels_
    unique_labels = set(labels)
    unique_labels.discard(-1)

    if len(unique_labels) == 0:
        return np.mean(coords, axis=0)

    best_label = max(unique_labels, key=lambda l: np.sum(labels == l))
    cluster_pts = coords[labels == best_label]
    return np.mean(cluster_pts, axis=0)


# ============================================================
# Evaluation
# ============================================================

def evaluate_clustering(all_signs, all_points, cluster_fn, sparse_only=False):
    """Evaluate a clustering method on all signs."""
    errors = []

    for sign_id, sign_data in all_signs.items():
        true_loc = sign_data.get('Geolocation', [])
        if len(true_loc) < 2:
            continue

        points = all_points.get(sign_id, [])
        if not points:
            continue

        # Sparse filter: only signs with <10 location points
        if sparse_only and len(points) >= 10:
            continue

        coords = to_projected(points)
        if len(coords) == 0:
            continue

        center = cluster_fn(coords)
        pred = from_projected(center)
        dist = geodesic(pred, true_loc).meters
        errors.append(dist)

    errors = np.array(errors)
    if len(errors) == 0:
        return {'mae': float('inf'), 'rmse': float('inf'),
                'r1m': 0, 'r2m': 0, 'n': 0}
    return {
        'mae': np.mean(errors),
        'rmse': np.sqrt(np.mean(errors ** 2)),
        'r1m': np.mean(errors < 1.0) * 100,
        'r2m': np.mean(errors < 2.0) * 100,
        'n': len(errors)
    }


def main():
    print("=" * 60)
    print("Table III Reproduction — NSAL Ablation Study")
    print("=" * 60)

    all_signs = load_all_signs()
    print(f"Loaded {len(all_signs)} signs")

    depth_data = np.load('./data/img_depth.npz')
    depth_dict = {key: depth_data[key] for key in depth_data.files}
    print(f"Loaded {len(depth_dict)} depth maps")

    # Pre-compute all location points (PlaneDepth) for every sign
    print("\n⏳ Computing location points for all signs...")
    t0 = time.time()
    all_points = {}
    for sign_id, sign_data in all_signs.items():
        boxes = sign_data.get('images', {})
        points = compute_location_points(sign_data, boxes, depth_dict)
        all_points[sign_id] = points
    print(f"   Done in {time.time() - t0:.1f}s")

    # Count sparse signs
    sparse_count = sum(1 for pts in all_points.values()
                       if 0 < len(pts) < 10)
    total_with_pts = sum(1 for pts in all_points.values() if len(pts) > 0)
    print(f"   Signs with points: {total_with_pts}")
    print(f"   Sparse signs (<10 pts): {sparse_count}")

    # --- Search best DBSCAN params for this context ---
    print("\n🔍 Searching DBSCAN hyperparameters...")
    best_mae = float('inf')
    best_eps, best_ms = 5.0, 2
    for eps in [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]:
        for ms in [1, 2, 3, 5]:
            r = evaluate_clustering(
                all_signs, all_points,
                lambda c, e=eps, m=ms: cluster_dbscan(c, eps=e, min_samples=m))
            if r['mae'] < best_mae:
                best_mae = r['mae']
                best_eps, best_ms = eps, ms
    print(f"   Best DBSCAN: eps={best_eps}, min_samples={best_ms}, MAE={best_mae:.2f}m")

    # --- Define all methods ---
    methods = {
        'AutoTS w/o MinCut': (cluster_no_mincut, False),
        'AutoTS w/o Weight': (cluster_no_weight, False),
        'AutoTS-K-Means':    (cluster_kmeans, False),
        'AutoTS-DBSCAN':     (lambda c: cluster_dbscan(c, eps=best_eps,
                              min_samples=best_ms), False),
        'AutoTS (ours)':     (cluster_nsal, False),
        'AutoTS-K-Means*':   (cluster_kmeans, True),
        'AutoTS-DBSCAN*':    (lambda c: cluster_dbscan(c, eps=best_eps,
                              min_samples=best_ms), True),
        'AutoTS*':           (cluster_nsal, True),
    }

    # --- Evaluate all methods ---
    print("\n" + "=" * 60)
    print("Evaluating all ablation variants...")
    print("=" * 60)

    results = {}
    for name, (fn, sparse) in methods.items():
        t0 = time.time()
        r = evaluate_clustering(all_signs, all_points, fn, sparse_only=sparse)
        r['time'] = time.time() - t0
        results[name] = r
        tag = " [SPARSE]" if sparse else ""
        print(f"  {name:<25} MAE={r['mae']:.2f}  RMSE={r['rmse']:.2f}  "
              f"R@1m={r['r1m']:.2f}%  R@2m={r['r2m']:.2f}%  "
              f"(n={r['n']})  {r['time']:.2f}s{tag}")

    # --- Save results ---
    os.makedirs('./results/v4_table3', exist_ok=True)
    csv_path = './results/v4_table3/results_table3.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'MAE(m)', 'RMSE(m)', 'R@1m(%)', 'R@2m(%)',
                         'N_signs', 'Time(s)'])
        for name, r in results.items():
            writer.writerow([name, f"{r['mae']:.2f}", f"{r['rmse']:.2f}",
                             f"{r['r1m']:.2f}", f"{r['r2m']:.2f}",
                             r['n'], f"{r['time']:.2f}"])
        writer.writerow([])
        writer.writerow(['--- Paper Reference ---'])
        paper = [
            ('AutoTS w/o MinCut', 2.51, 3.74, 27.11, 50.87),
            ('AutoTS w/o Weight', 2.49, 3.72, 28.81, 49.15),
            ('AutoTS-K-Means', 2.56, 3.82, 27.11, 47.46),
            ('AutoTS-DBSCAN', 2.48, 3.79, 28.81, 52.54),
            ('AutoTS (ours)', 2.38, 3.42, 30.51, 54.23),
            ('AutoTS-K-Means*', 2.46, 3.52, 28.81, 52.54),
            ('AutoTS-DBSCAN*', 2.42, 3.48, 30.51, 52.54),
            ('AutoTS*', 2.36, 3.37, 32.20, 54.23),
        ]
        for p in paper:
            writer.writerow([f"{p[0]} (paper)", p[1], p[2], p[3], p[4], '', ''])
    print(f"\n💾 Results saved to {csv_path}")

    # Save config
    with open('./results/v4_table3/config.txt', 'w') as f:
        f.write(f"DBSCAN best params: eps={best_eps}, min_samples={best_ms}\n")
        f.write(f"NSAL sigma: 2.5\n")
        f.write(f"Sparse threshold: <10 location points\n")
        f.write(f"Depth method: PlaneDepth (all variants)\n")

    # --- Comparison table ---
    print("\n" + "=" * 70)
    print("Table III Comparison (Ours vs Paper)")
    print("=" * 70)
    paper_dict = {p[0]: {'mae': p[1], 'rmse': p[2], 'r1m': p[3], 'r2m': p[4]}
                  for p in paper}
    print(f"{'Method':<25} {'MAE':>12} {'RMSE':>12} {'R@1m':>12} {'R@2m':>12}")
    print("-" * 70)
    for name in results:
        r = results[name]
        p = paper_dict.get(name, {})
        pm = f"({p.get('mae',0):.2f})" if p else ""
        pr = f"({p.get('rmse',0):.2f})" if p else ""
        p1 = f"({p.get('r1m',0):.2f})" if p else ""
        p2 = f"({p.get('r2m',0):.2f})" if p else ""
        print(f"{name:<25} {r['mae']:>5.2f}{pm:>7} "
              f"{r['rmse']:>5.2f}{pr:>7} "
              f"{r['r1m']:>5.2f}{p1:>7} "
              f"{r['r2m']:>5.2f}{p2:>7}")


if __name__ == '__main__':
    main()
