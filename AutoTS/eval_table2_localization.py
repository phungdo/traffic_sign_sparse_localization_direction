"""
eval_table2_localization.py — Reproduce Paper Table II
=====================================================
Compares 3 localization methods on KITTI-TS:
  1. GeoLocating [7]   — thin-lens depth + DBSCAN
  2. GeoLocating+NSAL  — thin-lens depth + NSAL (our sparse_cluster)
  3. AutoTS (ours)     — PlaneDepth depth + NSAL

Paper states: "the experimental results in localization are derived
from all the samples, rather than just the testing set."

Metrics: MAE(m), RMSE(m), R@1m(%), R@2m(%)
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
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from itertools import product

# ============================================================
# Camera parameters (KITTI default)
# ============================================================
# KITTI cam2 intrinsics (P2 matrix, standard across sequences):
#   f_x = 721.5377 px, c_x = 609.5593, image width = 1242
# The code uses CENTER_X = 621 (≈ 1242/2) and FOV_X = 45 degrees
FOCAL_LENGTH_PX = 721.5377
CENTER_X = 621
FOV_X = 45
OFFSET_X = 2
OFFSET_Y = 4

# German traffic sign standard heights (meters) by category
# Categories: 0-15 as defined in KITTI-TS
# Round signs (speed limits, prohibitions): diameter 0.60m
# Triangular signs (yield, warning): height ~0.63m (side=0.63)
# Rectangular signs: ~0.42m height
SIGN_HEIGHTS = {
    0: 0.60,   # 30 Zone (round)
    1: 0.60,   # 30 Zone End (round)
    2: 0.60,   # 60 Limit (round)
    3: 0.60,   # Construction (triangular, ~0.63 but using 0.60)
    4: 0.90,   # Yield (large triangle)
    5: 0.60,   # Pedestrian (round)
    6: 0.60,   # No Overtaking (round)
    7: 0.60,   # Mandatory Right (round)
    8: 0.60,   # No Entry (round)
    9: 0.42,   # One Way Right (rectangular)
    10: 0.42,  # One Way Left (rectangular)
    11: 0.42,  # No Parking (round, but small: 0.42)
    12: 0.60,  # Turn Left (round)
    13: 0.60,  # Priority (diamond-shaped, ~0.60)
    14: 0.60,  # Pedestrian Cross (triangular)
    15: 0.60,  # Curve Right (triangular)
}
DEFAULT_SIGN_HEIGHT = 0.60


def load_all_signs():
    """Load all 290 signs from train + test JSON."""
    with open('./KITTI-TS/train/sign_id_GT.json') as f:
        train = json.load(f)
    with open('./KITTI-TS/test/sign_id_GT.json') as f:
        test = json.load(f)
    all_signs = {**train, **test}
    return all_signs


# ============================================================
# Depth Estimation Methods
# ============================================================

def thin_lens_depth(box, category):
    """
    Thin-lens model: depth = f * H_real / h_pixel
    where f = focal length (pixels), H_real = real sign height (m),
    h_pixel = bounding box height in pixels.
    """
    pixel_height = box[3] - box[1]
    if pixel_height <= 0:
        return None
    real_height = SIGN_HEIGHTS.get(int(category), DEFAULT_SIGN_HEIGHT)
    depth = FOCAL_LENGTH_PX * real_height / pixel_height
    return depth


def planedepth_depth(box, img_depth):
    """
    PlaneDepth depth: extract median depth from depth map within bbox.
    Same as AutoTS code.
    """
    xmin = int(box[0]) + OFFSET_X
    ymin = int(box[1]) + OFFSET_Y
    xmax = int(box[2]) - OFFSET_X
    ymax = int(box[3]) - OFFSET_Y

    depth_map = img_depth[ymin:ymax, xmin:xmax]
    depth_values = depth_map[np.nonzero(depth_map)]
    if len(depth_values) == 0:
        return None
    return np.median(depth_values)


# ============================================================
# Location Point Calculation (shared)
# ============================================================

def get_geo(lat, lng, depth, alt):
    """Convert camera position + depth + bearing → sign GPS location."""
    if alt <= 90 and alt >= 0:
        alt = 90 - alt
    if alt > 90:
        alt = 450 - alt
    if alt < 0:
        alt = 90 - alt

    start = GeoPoint(latitude=lat, longitude=lng)
    destination = geo_distance.distance(meters=depth).destination(start, alt)
    return destination.latitude, destination.longitude


def compute_location_points(sign_data, boxes, depth_fn, depth_extra=None):
    """
    Compute location points for a single sign using a given depth function.
    Returns list of [lat, lon] points.
    """
    image_yaws = sign_data.get('image_yaws', {})
    geolocs = sign_data.get('image_geolocations', {})
    category = sign_data['category']

    if not image_yaws or not boxes:
        return []

    pos_list = []
    for obj_id, box in boxes.items():
        if depth_extra is not None:
            # PlaneDepth mode
            depth_map = depth_extra.get(obj_id, None)
            if depth_map is None:
                continue
            depth = planedepth_depth(box, depth_map)
        else:
            # Thin-lens mode
            depth = thin_lens_depth(box, category)

        if depth is None or depth <= 0:
            continue

        center_x = (box[0] + box[2]) / 2
        relative_angle = (center_x - CENTER_X) * (FOV_X / CENTER_X)
        yaw = np.degrees(image_yaws.get(obj_id, 0))
        alt = yaw - relative_angle

        lat0, lng0 = geolocs.get(obj_id, (None, None))
        if lat0 is None or lng0 is None:
            continue
        lat_, lng_ = get_geo(lat0, lng0, depth, alt)
        pos_list.append([lat_, lng_])

    return pos_list


# ============================================================
# Clustering Methods
# ============================================================

# Cache transformers as module-level singletons
_to_local = pyproj.Transformer.from_crs("epsg:4326", "epsg:3044", always_xy=True).transform
_to_geo = pyproj.Transformer.from_crs("epsg:3044", "epsg:4326", always_xy=True).transform


def point_transform(location_point):
    """WGS84 → local projected CRS (EPSG:3044)."""
    geom = Point(location_point)
    return transform(_to_local, geom)


def point_transform_back(geo_point):
    """Local projected CRS → WGS84."""
    geom = Point(geo_point)
    return transform(_to_geo, geom)


def gaussian_affinity_matrix(coords, sigma=2.5):
    """Vectorized Gaussian kernel affinity matrix."""
    sq_dists = cdist(coords, coords, metric='sqeuclidean')
    return np.exp(-sq_dists / (2 * sigma ** 2))


def nsal_cluster(points_list):
    """
    NSAL clustering (sparse_cluster from orientation_net.py).
    Returns predicted [lat, lon].
    """
    if not points_list or len(points_list) == 0:
        return None

    transformed = [point_transform(p) for p in points_list]
    coords = np.array([[p.x, p.y] for p in transformed])

    if len(coords) == 1:
        pred = point_transform_back(coords[0])
        return [pred.x, pred.y]

    # Sparse cluster (NSAL) — vectorized Gaussian kernel
    W = gaussian_affinity_matrix(coords)
    weight_sum = W.sum(axis=1)

    sorted_weights = np.sort(weight_sum)
    diffs = np.diff(sorted_weights)
    mean_diff = np.mean(diffs) if len(diffs) > 0 else 0

    count = 0
    for i, diff in enumerate(diffs):
        if diff > 3 * mean_diff:
            count += 1
        else:
            break

    num_del = min(int(len(coords) * 0.3), count)
    is_outlier = np.zeros(len(coords), dtype=bool)
    for _ in range(num_del):
        idx = np.argmin(weight_sum)
        is_outlier[idx] = True
        weight_sum[idx] = np.inf

    data_select = coords[~is_outlier]
    if len(data_select) == 0:
        center = np.mean(coords, axis=0)
    else:
        weights = W.sum(axis=1)[~is_outlier]
        weight_sum_total = np.sum(weights)
        if weight_sum_total == 0:
            center = np.mean(data_select, axis=0)
        else:
            norm_weights = weights / weight_sum_total * len(data_select)
            weighted = data_select * norm_weights[:, np.newaxis]
            center = np.mean(weighted, axis=0)

    pred = point_transform_back(center)
    return [pred.x, pred.y]


def dbscan_cluster(points_list, eps=5.0, min_samples=2):
    """
    DBSCAN clustering as used in GeoLocating [7].
    Returns predicted [lat, lon].
    """
    if not points_list or len(points_list) == 0:
        return None

    transformed = [point_transform(p) for p in points_list]
    coords = np.array([[p.x, p.y] for p in transformed])

    if len(coords) <= 2:
        center = np.mean(coords, axis=0)
        pred = point_transform_back(center)
        return [pred.x, pred.y]

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = clustering.labels_

    # Find largest cluster (excluding noise=-1)
    unique_labels = set(labels)
    unique_labels.discard(-1)

    if len(unique_labels) == 0:
        # All noise — use centroid
        center = np.mean(coords, axis=0)
    else:
        best_label = max(unique_labels, key=lambda l: np.sum(labels == l))
        cluster_points = coords[labels == best_label]
        center = np.mean(cluster_points, axis=0)

    pred = point_transform_back(center)
    return [pred.x, pred.y]


# ============================================================
# Evaluation
# ============================================================

def evaluate_method(all_signs, depth_mode, cluster_mode, depth_data=None,
                    dbscan_eps=5.0, dbscan_min_samples=2):
    """
    Run one method variant and compute metrics.
    depth_mode: 'thin_lens' or 'planedepth'
    cluster_mode: 'dbscan' or 'nsal'
    """
    errors = []

    # Pre-build depth dict once if planedepth mode
    img_depth = None
    if depth_mode == 'planedepth' and depth_data is not None:
        img_depth = {key: depth_data[key] for key in depth_data.files}

    for sign_id, sign_data in all_signs.items():
        true_loc = sign_data.get('Geolocation', [])
        if len(true_loc) < 2:
            continue

        boxes = sign_data.get('images', {})
        if not boxes:
            continue

        # Compute location points
        if depth_mode == 'planedepth':
            points = compute_location_points(sign_data, boxes, None, depth_extra=img_depth)
        else:
            points = compute_location_points(sign_data, boxes, thin_lens_depth)

        if not points:
            continue

        # Cluster
        if cluster_mode == 'nsal':
            pred = nsal_cluster(points)
        else:
            pred = dbscan_cluster(points, eps=dbscan_eps, min_samples=dbscan_min_samples)

        if pred is None:
            continue

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


def search_dbscan_hyperparams(all_signs):
    """Grid search for best DBSCAN eps and min_samples."""
    print("\n🔍 Searching DBSCAN hyperparameters...")
    best_mae = float('inf')
    best_params = (5.0, 2)

    eps_range = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
    min_samples_range = [1, 2, 3, 4, 5]

    for eps, ms in product(eps_range, min_samples_range):
        result = evaluate_method(all_signs, 'thin_lens', 'dbscan',
                                 dbscan_eps=eps, dbscan_min_samples=ms)
        if result['mae'] < best_mae:
            best_mae = result['mae']
            best_params = (eps, ms)
            print(f"  eps={eps:5.1f}, min_samples={ms}: "
                  f"MAE={result['mae']:.2f}m  RMSE={result['rmse']:.2f}m  "
                  f"R@1m={result['r1m']:.2f}%  R@2m={result['r2m']:.2f}%  "
                  f"(n={result['n']}) ← NEW BEST")

    print(f"\n✅ Best DBSCAN: eps={best_params[0]}, min_samples={best_params[1]}, MAE={best_mae:.2f}m")
    return best_params


def main():
    print("=" * 60)
    print("Table II Reproduction — Traffic Sign Localization")
    print("=" * 60)

    all_signs = load_all_signs()
    print(f"Loaded {len(all_signs)} signs")

    depth_data = np.load('./data/img_depth.npz')
    print(f"Loaded {len(depth_data.files)} depth maps")

    # --- Step 1: Search best DBSCAN params ---
    t0 = time.time()
    best_eps, best_ms = search_dbscan_hyperparams(all_signs)
    dbscan_search_time = time.time() - t0
    print(f"   DBSCAN search time: {dbscan_search_time:.1f}s")

    # --- Step 2: Evaluate all 3 methods ---
    print("\n" + "=" * 60)
    print("Evaluating all methods...")
    print("=" * 60)

    results = {}

    # Method 1: GeoLocating [7] — thin-lens + DBSCAN
    print("\n📍 Method 1: GeoLocating [7] (thin-lens + DBSCAN)")
    t0 = time.time()
    r = evaluate_method(all_signs, 'thin_lens', 'dbscan',
                        dbscan_eps=best_eps, dbscan_min_samples=best_ms)
    r['time'] = time.time() - t0
    r['time_per_sign'] = r['time'] / r['n'] * 1000  # ms
    results['GeoLocating [7]'] = r
    print(f"  MAE={r['mae']:.2f}m  RMSE={r['rmse']:.2f}m  "
          f"R@1m={r['r1m']:.2f}%  R@2m={r['r2m']:.2f}%  (n={r['n']})  "
          f"Time: {r['time']:.2f}s ({r['time_per_sign']:.1f}ms/sign)")

    # Method 2: GeoLocating+NSAL — thin-lens + NSAL
    print("\n📍 Method 2: GeoLocating+NSAL (thin-lens + NSAL)")
    t0 = time.time()
    r = evaluate_method(all_signs, 'thin_lens', 'nsal')
    r['time'] = time.time() - t0
    r['time_per_sign'] = r['time'] / r['n'] * 1000
    results['GeoLocating+NSAL'] = r
    print(f"  MAE={r['mae']:.2f}m  RMSE={r['rmse']:.2f}m  "
          f"R@1m={r['r1m']:.2f}%  R@2m={r['r2m']:.2f}%  (n={r['n']})  "
          f"Time: {r['time']:.2f}s ({r['time_per_sign']:.1f}ms/sign)")

    # Method 3: AutoTS (ours) — PlaneDepth + NSAL
    print("\n📍 Method 3: AutoTS (PlaneDepth + NSAL)")
    t0 = time.time()
    r = evaluate_method(all_signs, 'planedepth', 'nsal', depth_data=depth_data)
    r['time'] = time.time() - t0
    r['time_per_sign'] = r['time'] / r['n'] * 1000
    results['AutoTS (ours)'] = r
    print(f"  MAE={r['mae']:.2f}m  RMSE={r['rmse']:.2f}m  "
          f"R@1m={r['r1m']:.2f}%  R@2m={r['r2m']:.2f}%  (n={r['n']})  "
          f"Time: {r['time']:.2f}s ({r['time_per_sign']:.1f}ms/sign)")

    # --- Step 3: Save results ---
    os.makedirs('./results/v3_table2', exist_ok=True)

    csv_path = './results/v3_table2/results_table2.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'MAE(m)', 'RMSE(m)', 'R@1m(%)', 'R@2m(%)', 'N_signs', 'Time(s)', 'ms/sign'])
        for method, r in results.items():
            writer.writerow([method,
                             f"{r['mae']:.2f}",
                             f"{r['rmse']:.2f}",
                             f"{r['r1m']:.2f}",
                             f"{r['r2m']:.2f}",
                             r['n'],
                             f"{r.get('time', 0):.2f}",
                             f"{r.get('time_per_sign', 0):.1f}"])
        # Paper reference
        writer.writerow([])
        writer.writerow(['--- Paper Reference ---'])
        writer.writerow(['GeoLocating [7] (paper)', '3.98', '6.27', '11.86', '22.03', ''])
        writer.writerow(['GeoLocating+NSAL (paper)', '3.80', '5.92', '15.25', '27.12', ''])
        writer.writerow(['AutoTS (paper)', '2.38', '3.42', '30.51', '54.23', ''])

    print(f"\n💾 Results saved to {csv_path}")

    # Also save config
    config_path = './results/v3_table2/config.txt'
    with open(config_path, 'w') as f:
        f.write(f"DBSCAN best params: eps={best_eps}, min_samples={best_ms}\n")
        f.write(f"Focal length: {FOCAL_LENGTH_PX} px\n")
        f.write(f"Default sign height: {DEFAULT_SIGN_HEIGHT} m\n")
        f.write(f"Sign heights: {SIGN_HEIGHTS}\n")
        f.write(f"NSAL sigma: 2.5\n")
        f.write(f"CENTER_X: {CENTER_X}, FOV_X: {FOV_X}\n")
    print(f"💾 Config saved to {config_path}")

    # --- Print comparison table ---
    print("\n" + "=" * 60)
    print("Table II Comparison (Ours vs Paper)")
    print("=" * 60)
    paper = {
        'GeoLocating [7]':  {'mae': 3.98, 'rmse': 6.27, 'r1m': 11.86, 'r2m': 22.03},
        'GeoLocating+NSAL': {'mae': 3.80, 'rmse': 5.92, 'r1m': 15.25, 'r2m': 27.12},
        'AutoTS (ours)':    {'mae': 2.38, 'rmse': 3.42, 'r1m': 30.51, 'r2m': 54.23},
    }
    print(f"{'Method':<25} {'MAE(m)':>8} {'RMSE(m)':>9} {'R@1m(%)':>9} {'R@2m(%)':>9}")
    print("-" * 60)
    for method in results:
        r = results[method]
        p = paper[method]
        print(f"{method:<25} {r['mae']:>6.2f}({p['mae']:.2f}) "
              f"{r['rmse']:>6.2f}({p['rmse']:.2f}) "
              f"{r['r1m']:>6.2f}({p['r1m']:.2f}) "
              f"{r['r2m']:>6.2f}({p['r2m']:.2f})")


if __name__ == '__main__':
    main()
