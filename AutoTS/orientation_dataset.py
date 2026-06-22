import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from torchvision import models, transforms
from sklearn.cluster import AgglomerativeClustering
import cv2
from tqdm import tqdm

from geopy import Point
from geopy import distance

class TSDataset(Dataset):
    def __init__(self, data_path, img_root, device,
                 config_path, weight_path, training=False):

        self.image_root = img_root
        self.device = device
        self.training = training

        self.detector = self._init_detector(config_path, weight_path)
        self.img_feature_extractor, self.transform = self._init_extractor()

        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.keys = list(self.data.keys())

        self.img_dict = {}
        self.ids_dict = {}
        self.box_dict = {}
        self.category_dict = {}
        self._load_images_and_boxes()

        self.image_feature = self.get_img_feature()
        self.roi_feature, self.detected_boxes = self.get_roi_feature()
        self.sift_feature = self.get_SIFT_feature()

        depth_data = np.load('./data/img_depth.npz')
        if self.training:
            self.location_points = self.get_location_points(depth_data)
        else:
            self.location_points = self.get_location_points(depth_data)

    def _load_images_and_boxes(self):

        for key in tqdm(self.keys, desc="Loading images and boxes"):
            imgs, ids, boxes = {}, list(self.data[key]['images'].keys()), {}
            for frame_id, bbox in self.data[key]['images'].items():
                img_path = os.path.join(self.image_root, f"{frame_id}.png")
                image = np.array(Image.open(img_path).convert("RGB"))
                imgs[frame_id] = image
                boxes[frame_id] = bbox

            self.img_dict[key] = imgs
            self.ids_dict[key] = ids
            self.box_dict[key] = boxes
            self.category_dict[key] = self.data[key]['category']

    def _init_detector(self, config_path, weight_path):
        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        cfg.MODEL.WEIGHTS = weight_path
        cfg.MODEL.DEVICE = self.device
        predictor = DefaultPredictor(cfg)
        return predictor

    def _init_extractor(self, resize=224):
        resnet_model = models.resnet101(pretrained=True)
        resnet_model = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))
        resnet_model = resnet_model.to(self.device)
        transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return resnet_model, transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):

        key = self.keys[idx]
        image_feature = self.image_feature[key]
        roi_feature = self.roi_feature[key]
        sift_feature = self.sift_feature[key]
        direction = int(self.data[key]['direction'])
        geolocation = torch.tensor(self.data[key]['Geolocation'])
        category = int(self.data[key]['category'])
        location_point = self.location_points[key]

        return {
            'image_feature': image_feature,
            'roi_feature': roi_feature,
            'sift_feature': sift_feature,
            'direction': direction,
            'geolocation': geolocation,
            'location_point': location_point,
            'category': category
        }


    def get_img_feature(self):

        img_featrues = {}
        for key in self.img_dict.keys():
            images = list(self.img_dict[key].values())
            imgs = [self.transform(Image.fromarray(img)) for img in images]
            imgs = torch.stack(imgs, dim=0).to(self.device)
            with torch.no_grad():
                features = self.img_feature_extractor(imgs)
                features = features.view(features.size(0), -1)
            img_featrues[key] = features

        return img_featrues

    def get_roi_feature(self):

        roi_feature_dict = {}
        detected_boxes_dict = {}
        for key in self.img_dict.keys():
            images = list(self.img_dict[key].values())
            ids = self.ids_dict[key]
            boxes_g = self.box_dict[key]
            category = int(self.category_dict[key])
            instances_roi = {}
            for i in range(len(images)):
                with torch.no_grad():
                    outputs, roi_features = self.detector(images[i])
                instances = outputs[0]['instances'].to("cpu")

                boxes = instances.pred_boxes.tensor.numpy().tolist()
                classes = instances.pred_classes.numpy().tolist()

                instances_roi[ids[i]] = {
                    "boxes": boxes,
                    "classes": classes,
                    "roi_features": roi_features
                }

            roi_features = []
            detected_boxes = {}
            for img in ids:
                if len(roi_features) == 0:
                    for j in range(len(instances_roi[img]['boxes'])):
                        if (self.IoU(boxes_g[img], instances_roi[img]['boxes'][j]) > 0.5) and (
                                category == instances_roi[img]['classes'][j]):
                            roi_features.append(instances_roi[img]['roi_features'][j])
                            detected_boxes[img] = instances_roi[img]['boxes'][j]
                            break
                    continue
                if len(roi_features) >= 1:
                    for j in range(len(instances_roi[img]['boxes'])):
                        if (self.IoU(list(detected_boxes.values())[-1], instances_roi[img]['boxes'][j]) > 0.5) and (
                                category == instances_roi[img]['classes'][j]):
                            roi_features.append(instances_roi[img]['roi_features'][j])
                            detected_boxes[img] = instances_roi[img]['boxes'][j]
                            break

            if len(roi_features) == 0:
                roi_features.append(np.zeros(1024))
            roi_features = [torch.from_numpy(arr).float().to(self.device) for arr in roi_features]

            if self.training:
                detected_boxes = boxes_g
            roi_feature_dict[key] = roi_features
            detected_boxes_dict[key] = detected_boxes

        return roi_feature_dict, detected_boxes_dict

    def IoU(self, box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2

        inter_x1 = max(x1, x3)
        inter_y1 = max(y1, y3)
        inter_x2 = min(x2, x4)
        inter_y2 = min(y2, y4)

        intersection_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)
        box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
        box2_area = (x4 - x3 + 1) * (y4 - y3 + 1)

        iou = intersection_area / float(box1_area + box2_area - intersection_area)

        return iou

    def get_SIFT_feature(self):
        sift_feature_dict = {}
        for key in self.img_dict.keys():
            descriptors_list = []
            images = self.img_dict[key]
            boxes = self.detected_boxes[key]
            if len(list(boxes.keys())) == 0:
                sift_feature_dict[key] = [torch.zeros(160, dtype=torch.float32).to(self.device)]
                continue
            for id, box in boxes.items():
                xmin, ymin, xmax, ymax = map(int, box)
                roi = images[id][ymin:ymax, xmin:xmax]

                last_box = boxes[list(boxes.keys())[-1]]
                scale = (last_box[3] - last_box[1]) / (ymax - ymin)
                new_h = int(scale * (ymax - ymin))
                new_w = int(scale * (xmax - xmin))
                roi_resized = cv2.resize(roi, (new_w, new_h))
                descriptors = self.extract_sift_desciptors(roi_resized)
                if descriptors is not None:
                    descriptors_list.append(descriptors)

            if len(descriptors_list) == 0:
                sift_fea = np.zeros((1, 160))
            else:
                sift_fea = self.sift_to_vec(descriptors_list)
            sift_feature = [torch.tensor(vec, dtype=torch.float32).to(self.device) for vec in sift_fea]
            sift_feature_dict[key] = sift_feature
        return sift_feature_dict

    def extract_sift_desciptors(self, image):

        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is None:
            return None

        h, w = image.shape[:2]
        descriptors_loc = []
        mean_ = np.mean(np.nonzero(descriptors))
        for i, (kp, desc) in enumerate(zip(keypoints, descriptors)):
            x, y = kp.pt
            x_norm = x / w * mean_
            y_norm = y / h * mean_
            pos_encoding = np.array([x_norm] * 5 + [y_norm] * 5)
            full_descriptor = np.concatenate((pos_encoding, desc))
            descriptors_loc.append(full_descriptor)

        descriptors_loc = np.array(descriptors_loc, dtype=np.float32)
        norms = np.linalg.norm(descriptors_loc, axis=1, keepdims=True) + 1e-8
        descriptors_loc = descriptors_loc / norms
        return descriptors_loc

    def sift_to_vec(self, descriptor_groups):

        descriptors = np.vstack(descriptor_groups)
        if len(descriptors) <= 1:
            return np.zeros((1, 160), dtype=np.float32)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.6,
            linkage='ward'
        )
        clustering.fit(descriptors)
        labels = clustering.labels_

        img_features = np.zeros((len(descriptor_groups), 160), "float32")
        count = 0
        for i in range(len(descriptor_groups)):
            for j in range(len(descriptor_groups[i])):
                img_features[i][labels[j+count]] += 1
            count += len(descriptor_groups[i])

        return img_features

    def get_location_points(self, depth_data):

        CENTER_X = 621
        FOV_X = 45
        OFFSET_X = 2
        OFFSET_Y = 4
        img_depth = {key: depth_data[key] for key in depth_data.files}

        position_dict = {}

        for key, boxes in self.box_dict.items():
            image_yaws = self.data[key].get('image_yaws', {})
            geolocs = self.data[key].get('image_geolocations', {})

            if not image_yaws or not boxes:
                position_dict[key] = []
                continue

            pos_list = []
            for obj_id, box in boxes.items():
                xmin = int(box[0]) + OFFSET_X
                ymin = int(box[1]) + OFFSET_Y
                xmax = int(box[2]) - OFFSET_X
                ymax = int(box[3]) - OFFSET_Y

                depth_crop = img_depth.get(obj_id, None)
                depth_map = depth_crop[ymin:ymax, xmin:xmax]
                depth_values = depth_map[np.nonzero(depth_map)]
                depth = np.median(depth_values)

                center_x = (box[0] + box[2]) / 2
                relative_angle = (center_x - CENTER_X) * (FOV_X / CENTER_X)
                yaw = np.degrees(image_yaws.get(obj_id, 0))
                alt = yaw - relative_angle

                lat0, lng0 = geolocs.get(obj_id, (None, None))
                lat_, lng_ = self.get_GEO(lat0, lng0, depth, alt)
                pos_list.append([lat_, lng_])

            position_dict[key] = pos_list

        return position_dict


    def get_GEO(self, lat, lng, depth, alt):

        if alt <= 90 and alt >= 0:
            alt = 90 - alt
        if alt > 90:
            alt = 450 - alt
        if alt < 0:
            alt = 90 - alt

        start = Point(latitude=lat, longitude=lng)
        destination = distance.distance(meters=depth).destination(start, alt)

        return destination.latitude, destination.longitude