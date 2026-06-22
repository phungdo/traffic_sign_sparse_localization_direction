import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances
from shapely.ops import transform
from shapely.geometry import Point
import pyproj
from geopy.distance import geodesic

class OrientationNet(nn.Module):
    def __init__(self,
                 roi_input_size, sift_input_size, roi_hidden, sift_hidden, img_hidden,
                 num_classes,
                 num_layers=2, d_model=128, nhead=4, device="cuda"):
        super().__init__()

        self.device = device
        self.roi_fc = nn.Linear(roi_input_size, roi_hidden)
        self.sift_fc = nn.Linear(sift_input_size, sift_hidden)
        self.image_fc = nn.Linear(2048, img_hidden)


        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )

        self.out_fc = nn.Linear(d_model + img_hidden, num_classes)

    def forward(self, data):
        img_input = data['image_feature']
        roi_input = data['roi_feature']
        sift_input = data['sift_feature']
        max_seq_len = max(len(sample) for sample in img_input)
        roi_input = self.pad_tensor_list_batch(roi_input, max_seq_len)
        sift_input = self.pad_tensor_list_batch(sift_input, max_seq_len)

        roi_emb = self.roi_fc(roi_input)
        sift_emb = self.sift_fc(sift_input)

        orient_feas = torch.cat((roi_emb, sift_emb), dim=2)
        orient_feas = self.transformer_encoder(orient_feas)
        seq_feature = orient_feas.mean(dim=1)

        img_feas = []
        for img in img_input:
            img = self.image_fc(img).mean(dim=0)
            img_feas.append(img)
        img_feas = torch.stack(img_feas, dim=0)

        fused = torch.cat((seq_feature, img_feas), dim=1)
        orientation_logits = self.out_fc(fused)


        return orientation_logits

    def compute_loss(self, logits, targets, weight):
        loss = F.cross_entropy(logits, targets, weight=weight)
        return loss

    def pad_tensor_list_batch(self, features, max_seq_len, pad_value=0.0):
        batch_size = len(features)
        feature_dim = features[0][0].shape[0]
        padded_tensor = torch.full((batch_size, max_seq_len, feature_dim), pad_value).to(self.device)

        for i, sample in enumerate(features):
            for j, tensor in enumerate(sample):
                padded_tensor[i, j] = tensor

        return padded_tensor


def TS_localization(geo_data):
    errors = []

    for sample in geo_data:
        points = sample['location_point']
        true_loc = sample['geolocation']

        if not points:
            continue

        transformed_points = [point_transfor(p) for p in points]
        coords = np.array([[p.x, p.y] for p in transformed_points])

        cluster_result = sparse_cluster(coords)

        pred_center = point_transfor_back(cluster_result)
        center_lat, center_lon = pred_center.x, pred_center.y

        distance_m = geodesic([center_lat, center_lon], true_loc).meters
        errors.append(distance_m)

    errors = np.array(errors)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    recall_1m = np.mean(errors < 1.0)
    recall_2m = np.mean(errors < 2.0)

    return errors, mae, rmse, recall_1m, recall_2m


def sparse_cluster(data):
    W = pairwise_distances(data, metric=weight)
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

    num_del = min(int(len(data) * 0.3), count)

    is_outlier = np.zeros(len(data), dtype=bool)
    for _ in range(num_del):
        idx = np.argmin(weight_sum)
        is_outlier[idx] = True
        weight_sum[idx] = np.inf

    data_select = data[~is_outlier]
    if len(data_select) == 0:
        return np.mean(data, axis=0)

    weights = W.sum(axis=1)[~is_outlier]
    weight_sum_total = np.sum(weights)

    if weight_sum_total == 0:
        return np.mean(data_select, axis=0)

    norm_weights = weights / weight_sum_total * len(data_select)
    weighted = data_select * norm_weights[:, np.newaxis]
    center = np.mean(weighted, axis=0)

    return center


def weight(x, y, sigma=2.5):
    return np.exp(-1.0 * (x - y).T @ (x - y) /(2 * sigma**2))


def point_transfor(location_point):
    transformer_to_local = pyproj.Transformer.from_crs("epsg:4326", "epsg:3044", always_xy=True).transform
    geom = Point(location_point)

    return transform(transformer_to_local, geom)

def point_transfor_back(GEO_point):
    transformer_to_geo = pyproj.Transformer.from_crs("epsg:3044", "epsg:4326", always_xy=True).transform
    geom = Point(GEO_point)

    return transform(transformer_to_geo, geom)