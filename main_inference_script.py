import os
import time
# Start measuring time
total_start_time, libraries_start_time = time.time(), time.time()
from scipy.stats import linregress
import json
import random
import cv2
import numpy as np
from shapely import Polygon
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import yaml
import argparse
import gc
from mayavi import mlab
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from compute_3D_box_Kitti import draw_boxes_on_image, load_ground_truth
from dataset_preparation import read_calib_file, project_lidar_to_image
from yolov8n_segmentation import YOLOv8nSegmentation
from frustum_pointnet_model_v2 import FrustumPointNet

libraries_end_time = time.time()
print(f'-' * 50)
print(f"Libraries loaded in {(libraries_end_time - libraries_start_time) * 1000:.1f}ms.")
print("-" * 50)

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('class_mappings.json', 'r') as f:
    mappings = json.load(f)
YOLO_TO_KITTI = mappings['YOLO_TO_KITTI']
CLS_TO_NAME = mappings['CLS_TO_NAME']
Frustum_TO_NAME = mappings['Frustum_TO_NAME']

def map_yolo_to_kitti(yolo_class_id):
    return CLS_TO_NAME.get(str(yolo_class_id), 'Misc')

def print_separator(title=None):
    if title:
        print("\n" + "-" * 10 + f" {title} " + "-" * 10)
    else:
        print("\n" + "-" * 50)
        
def load_model(model_path):
    print_separator("Model Loading")
    model_start_time = time.time()
    model = FrustumPointNet(num_classes=9)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model_end_time = time.time()
    print(f"Frustum PointNet model loaded in {(model_end_time - model_start_time) * 1000:.1f} ms.")
    model.eval()
    return model

def voxel_grid_downsampling(points, voxel_size):
    coords = ((points[:, :3] - points[:, :3].min(axis=0)) // voxel_size).astype(np.int32)
    _, unique_indices = np.unique(coords, axis=0, return_index=True)
    return points[unique_indices]

def preprocess_lidar_data(point_cloud):
    preprocess_start_time = time.time()
    point_cloud = voxel_grid_downsampling(point_cloud, voxel_size=0.1)
    preprocess_end_time = time.time()
    print(f"\nPreprocessing of LiDAR points completed in {(preprocess_end_time - preprocess_start_time) * 1000:.1f} ms.")
    return point_cloud

def preprocess_lidar_data_frustum(point_cloud, x_limits=(0, 150), y_limits=(-40, 40), z_limits=(-2, 5)):
    mask = (point_cloud[:, 0] >= x_limits[0]) & (point_cloud[:, 0] <= x_limits[1]) & \
           (point_cloud[:, 1] >= y_limits[0]) & (point_cloud[:, 1] <= y_limits[1]) & \
           (point_cloud[:, 2] >= z_limits[0]) & (point_cloud[:, 2] <= z_limits[1])
    return point_cloud[mask]

def calculate_bbox_size(points: np.ndarray) -> tuple:
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    size = max_coords - min_coords
    return tuple(size)

def calculate_centroid(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        print("No points found.")
        return np.array([0, 0, 0])
    return np.mean(points, axis=0)

def calculate_distance_to_convex_hull(points: np.ndarray) -> float:
    if len(points) < 4:
        distances = np.linalg.norm(points, axis=0)
        return np.min(distances)
    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        distances = np.linalg.norm(hull_points, axis=1)
        return np.min(distances)
    except Exception as e:
        print(f"Error computing convex hull: {e}")
        distances = np.linalg.norm(points, axis=1)
        return np.min(distances)

def calculate_distance(centroid: np.ndarray) -> float:
    return np.linalg.norm(centroid)

def estimate_heading_angle(points):
    if points.shape[0] < 2:
        print("Not enough points to estimate heading angle.")
        return 0.0
    projected_points = points[:, [0, 2]]
    pca = PCA(n_components=2)
    pca.fit(projected_points)
    principal_axis = pca.components_[0]

    rotation_y = np.arctan2(principal_axis[0], principal_axis[1])
    return round(rotation_y, 2)
    
def generate_frustum_proposals(image, point_cloud, calib, model, filename, split):
    segmentation_model = YOLOv8nSegmentation('yolov8n-seg.pt')
    results = segmentation_model.segment(image)
    distances, centroids, frustum_points_list, frustum_labels_list, bounding_boxes_list, confidences, bbox_sizes, dt_rys = [], [], [], [], [], [], [], []
    valid_frustum_indices = []
    bbox_info_list = []

    for result in results:
        segmented_image = result.plot()
        if result.masks == None :
            print("No objects detected in the image.")
            continue
    
        pts_3d_cam = project_lidar_to_image(calib, point_cloud)

        for i, (box, conf, cls, mask) in enumerate(zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.masks.data.cpu().numpy())):
            if conf < config['parameters']['confidence_threshold']:
                continue

            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            mask = (pts_3d_cam[:, 0] >= x1) & (pts_3d_cam[:, 0] <= x2) & \
                   (pts_3d_cam[:, 1] >= y1) & (pts_3d_cam[:, 1] <= y2)

            frustum_points = point_cloud[mask]
            z_min, z_max = -1.5, 10
            height_mask = (frustum_points[:, 2] > z_min) & (frustum_points[:, 2] < z_max)
            frustum_points = frustum_points[height_mask]
            if len(frustum_points) > 0:
                clustering = DBSCAN(eps=config['parameters']['eps'], min_samples=config['parameters']['min_samples']).fit(frustum_points[:, :3])
                labels = clustering.labels_
                unique_labels, counts = np.unique(labels, return_counts=True)
                largest_cluster = unique_labels[np.argmax(counts)]
                cluster_mask = labels == largest_cluster
                frustum_points = frustum_points[cluster_mask]
                points_tensor = torch.tensor(frustum_points[:, :3], dtype=torch.float32).transpose(0, 1).unsqueeze(0)
                with torch.no_grad():
                    outputs = model(points_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    confidences_tensor, predicted_class = torch.max(probabilities, 1)
                confidences.append(confidences_tensor.item())
                pred_class = predicted_class.item()
                centroid = calculate_centroid(frustum_points)
                if len(frustum_points) >= 4:
                    distance = calculate_distance_to_convex_hull(frustum_points)
                else:
                    print(f"Not enough points to compute convex hull for object {i}.")
                    distance = calculate_distance(centroid[:3])
                bbox_size = calculate_bbox_size(frustum_points)
                dt_ry = estimate_heading_angle(frustum_points)
                centroids.append(centroid)
                distances.append(distance)
                bbox_sizes.append(bbox_size)
                dt_rys.append(dt_ry)
                yolocls = YOLO_TO_KITTI.get(str(int(cls)), 7)
                kitti_class = map_yolo_to_kitti(yolocls)
                pred_class_name = Frustum_TO_NAME.get(str(pred_class), 7)
                print("\n" + "-" * 10 + f" Detection {i + 1} " + "-" * 10)
                print(f"Detected object: {kitti_class}")
                print(f" - YOLO class ID: {cls}")
                print(f' - YOLO class name: {kitti_class}')
                print(f" - Frustum class ID: {pred_class}")
                print(f" - Frustum class name: {pred_class_name}")
                print(f" - Confidence: {conf:.2%}")
                print(f" - Distance from sensor: {distance:.2f} m")
                print("-" * 50)
                frustum_points_list.append(frustum_points)
                frustum_labels_list.append(pred_class)
                valid_frustum_indices.append(len(frustum_points_list))
                bounding_boxes_list.append((x1, x2, y1, y2))
                bbox_info_list.append({'cls': cls, 'conf': conf})

    return frustum_points_list, frustum_labels_list, bounding_boxes_list, distances, segmented_image, bbox_sizes, dt_rys

def project_to_image(corners_3d: np.ndarray, calib) -> np.ndarray:
    P = calib['P2']
    R0_rect = calib['R0_rect']
    Tr_velo_to_cam = calib['Tr_velo_to_cam']

    corners_3d_hom = np.hstack((corners_3d, np.ones((corners_3d.shape[0], 1))))
    corners_cam = np.dot(R0_rect, np.dot(Tr_velo_to_cam, corners_3d_hom.T)).T
    corners_cam_hom = np.hstack((corners_cam, np.ones((corners_cam.shape[0], 1))))
    corners_img_hom = np.dot(P, corners_cam_hom.T).T
    corners_img = corners_img_hom[:, :2] / corners_img_hom[:, 2].reshape(-1, 1)
    
    return corners_img

def project_3d_bbox(image, bbox_center: np.ndarray, bbox_size: tuple, calib, color) -> np.ndarray:
    bbox_center = np.array(bbox_center)
    corners_3d = get_3d_box_corners(bbox_center, bbox_size)
    corners_2d = project_to_image(corners_3d, calib)

    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    
    for edge in edges:
        pt1 = tuple(corners_2d[edge[0]].astype(int))
        pt2 = tuple(corners_2d[edge[1]].astype(int))
        cv2.line(image, pt1, pt2, color, 2)
    
    return image

def compare_distances(detected_distances, gt_distances, obj):
    error = abs(detected_distances - gt_distances)
    print(f"\nDistance Comparison for Object '{obj}':")
    print(f" - Detected Distance: {detected_distances:.2f} m")
    print(f" - Ground Truth Distance: {gt_distances:.2f} m")
    print(f" - Error: {error:.2f} m")

def compute_3d_box_volume(corners):
    edge1 = np.linalg.norm(corners[0] - corners[1])
    edge2 = np.linalg.norm(corners[1] - corners[2])
    edge3 = np.linalg.norm(corners[0] - corners[4])

    return edge1 * edge2 * edge3


def compute_iou_3d(corners1, corners2):
    vol1 = compute_3d_box_volume(corners1)
    vol2 = compute_3d_box_volume(corners2)

    base_corners1 = corners1[[4, 5, 6, 7], :]
    base_corners2 = corners2[[4, 5, 6, 7], :]

    def order_polygon(corners):
        center = np.mean(corners[:, [0, 2]], axis=0)
        angles = np.arctan2(corners[:, 2] - center[1], corners[:, 0] - center[0])
        sort_order = np.argsort(angles)
        return corners[sort_order]

    poly1_pts = order_polygon(base_corners1)
    poly2_pts = order_polygon(base_corners2)

    poly1 = Polygon(poly1_pts[:, [0, 2]])
    poly2 = Polygon(poly2_pts[:, [0, 2]])

    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    inter_area = poly1.intersection(poly2).area

    if inter_area == 0:
        return 0.0

    ymin1 = np.min(corners1[:, 1])
    ymax1 = np.max(corners1[:, 1])
    ymin2 = np.min(corners2[:, 1])
    ymax2 = np.max(corners2[:, 1])

    inter_ymin = max(ymin1, ymin2)
    inter_ymax = min(ymax1, ymax2)
    inter_h = max(0.0, inter_ymax - inter_ymin)

    if inter_h == 0:
        return 0.0

    inter_vol = inter_area * inter_h

    union_vol = vol1 + vol2 - inter_vol

    iou = inter_vol / union_vol if union_vol > 0 else 0.0
    return iou

def transform_point_to_cam(point_lidar, calib):
    R0_rect = calib['R0_rect']
    Tr_velo_to_cam = calib['Tr_velo_to_cam']

    R = Tr_velo_to_cam[:, :3]
    T = Tr_velo_to_cam[:, 3]

    point_cam = R0_rect @ (R @ point_lidar[:3] + T)

    return point_cam

import matplotlib.pyplot as plt

def match_detections_to_ground_truth_centroid(detections, ground_truths, distance_threshold=2.0):
    matches = []
    unmatched_detections = list(range(len(detections)))
    unmatched_ground_truths = list(range(len(ground_truths)))

    distance_matrix = np.full((len(detections), len(ground_truths)), np.inf)

    for i, detection in enumerate(detections):
        for j, gt in enumerate(ground_truths):
            if detection['class'] != gt['class']:
                continue
            dist = np.linalg.norm(detection['center'] - gt['center'])
            distance_matrix[i, j] = dist

    while True:
        min_idx = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
        min_distance = distance_matrix[min_idx]
        if min_distance > distance_threshold:
            break
        i, j = min_idx
        matches.append((i, j, min_distance))
        distance_matrix[i, :] = np.inf
        distance_matrix[:, j] = np.inf
        unmatched_detections.remove(i)
        unmatched_ground_truths.remove(j)

    return matches, unmatched_detections, unmatched_ground_truths

def generate_points_on_bbox(corners_3d_cam, calib, num_points=1000):
    corners_3d_lidar = transform_corners_to_lidar(corners_3d_cam, calib)

    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [1, 3, 7, 5],
        [0, 2, 6, 4]
    ]

    points = []

    for face in faces:
        face_corners = corners_3d_lidar[face]
        for _ in range(num_points // 6):
            u, v = np.random.rand(2)
            if u + v > 1:
                u, v = 1 - u, 1 - v
            point = (1 - u - v) * face_corners[0] + u * face_corners[1] + v * face_corners[3]
            points.append(point)

    return np.array(points)

def run_inference(image_path, point_cloud_path, calib_path, label_path, model, filename, split, total_start_time, mse_compute=False):
    try:
        load_data_start_time = time.time()
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        point_cloud = np.fromfile(point_cloud_path, dtype=np.float32).reshape(-1, 4)
        calib = read_calib_file(calib_path)
        load_data_end_time = time.time()
        print_separator("Data Loading")
        print(f"Data loading completed in {(load_data_end_time - load_data_start_time) * 1000:.1f} ms.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    point_cloud_frustum = preprocess_lidar_data_frustum(point_cloud[:, :3])
    f_points, f_labels, b_boxes, distances, segmented_image, bbox_sizes, dt_rys = generate_frustum_proposals(image, point_cloud_frustum, calib, model, filename, split)
    print(f"\nTotal number of detected objects: {len(f_points)}")

    centroids = []
    per_class_data = {}
    for i, (box, dist) in enumerate(zip(b_boxes, distances)):
        x1, x2, y1, y2 = box
        label = f"{map_yolo_to_kitti(f_labels[i])} ID:{i} Dist:{dist:.2f}m"
        color = (int(random.random()*255), int(random.random()*255), int(random.random()*255))
        cv2.putText(image, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        bbox_center = calculate_centroid(f_points[i])
        centroids.append(bbox_center)
        bbox_size = bbox_sizes[i]
        segmented_image = project_3d_bbox(image, bbox_center, bbox_size, calib, color)
        for point in f_points[i]:
            point_cam = project_lidar_to_image(calib, point.reshape(1, -1))
            cv2.circle(segmented_image, (int(point_cam[0, 0]), int(point_cam[0, 1])), 2, (0, 0, 255), -1)
    if mse_compute == False:
        cv2.imwrite(f'dataset/kitti/{split}/segmented_images/{filename}_3D_box.png', segmented_image)

    processed_point_cloud = preprocess_lidar_data(point_cloud)

    detection_list, dt_ry_cams = [], []
    print_separator("Detection Details")
    for i, (dt_center, dt_size, dt_ry) in enumerate(zip(centroids, bbox_sizes, dt_rys)):
        dt_ry_cam = dt_ry - np.pi / 2
        dt_ry_cams.append(dt_ry_cam)
        dt_center_cam = transform_point_to_cam(dt_center, calib)
        dt_corners = compute_box_3d(dt_size, dt_center_cam, dt_ry_cam)
        detection_list.append({
            'index': i,
            'class': map_yolo_to_kitti(f_labels[i]),
            'center': dt_center_cam,
            'size': dt_size,
            'rotation_y': dt_ry_cam,
            'corners': dt_corners,
            'distance': distances[i]
        })
        print(f"\nDetected Object {i + 1} Details:")
        print(f" - Dimensions:")
        print(f"   - Length: {dt_size[0]:.2f} m")
        print(f"   - Width: {dt_size[1]:.2f} m")
        print(f"   - Height: {dt_size[2]:.2f} m")
        print(f" - Position:")
        print(f"   - x: {dt_center_cam[2]:.2f} m")
        print(f"   - y: {dt_center_cam[1]:.2f} m")
        print(f"   - z: {dt_center_cam[0]:.2f} m")
        print(f" - Rotation (ry): {dt_ry:.2f} radians")
        print(f" - Rotation cam (ry_cam): {dt_ry_cam}")

    image = cv2.imread(image_path)
    labels = None
    if label_path:
        labels = load_ground_truth(label_path)
        print(f"Loaded {len(labels)} ground truth label(s).")
    else:
        print("No ground truth labels found.")
    if labels:
        if mse_compute == False:
            gt_image = draw_boxes_on_image(image, calib_path, label_path)
            cv2.imwrite(f'dataset/kitti/{split}/segmented_images/{filename}_compare.png', gt_image)

        print_separator("Ground Truth Object Details")
        for gt_label in labels:
            if gt_label['type'] == 'DontCare' or gt_label['type'] == 'Misc':
                continue
            print("\nGround Truth Object Details:")
            print(f" - Type: {gt_label['type']}")
            print(f" - Dimensions:")
            print(f"   - Length: {gt_label['dimensions'][0]:.2f} m")
            print(f"   - Width: {gt_label['dimensions'][1]:.2f} m")
            print(f"   - Height: {gt_label['dimensions'][2]:.2f} m")
            print(f" - Position:")
            print(f"   - x: {gt_label['location'][0]:.2f} m")
            print(f"   - y: {gt_label['location'][1]:.2f} m")
            print(f"   - z: {gt_label['location'][2]:.2f} m")
            print(f" - Rotation (ry): {gt_label['rotation_y']:.2f} radians")
    
        ground_truth_list = []
        for j, gt_label in enumerate(labels):
            if gt_label['type'] == 'DontCare' or gt_label['type'] == 'Misc':
                continue
            gt_dimensions = gt_label['dimensions']
            gt_location = gt_label['location']
            gt_rotation_y = gt_label['rotation_y']
            gt_size = (gt_dimensions[2], gt_dimensions[1], gt_dimensions[0])
            gt_center = np.array(gt_location)
            gt_corners = compute_box_3d(gt_dimensions, gt_location, gt_rotation_y)
            gt_points = generate_points_on_bbox(gt_corners, calib)

            ground_truth_list.append({
                'index': j,
                'class': gt_label['type'],
                'center': gt_center,
                'size': gt_size,
                'rotation_y': gt_rotation_y,
                'corners': gt_corners,
                'points': gt_points
            })
        
        for detection in detection_list:
            class_name = detection['class']
            if class_name not in per_class_data:
                per_class_data[class_name] = {
                    'detected_distances': [],
                    'ground_truth_distances': [],
                    'distance_errors': [],
                    'matches': 0,
                    'detections': 0,
                    'ground_truths': 0,
                    'false_positives': 0,
                    'false_negatives': 0
                }
            per_class_data[class_name]['detections'] += 1

        for gt in ground_truth_list:
            class_name = gt['class']
            if class_name not in per_class_data:
                per_class_data[class_name] = {
                    'detected_distances': [],
                    'ground_truth_distances': [],
                    'distance_errors': [],
                    'matches': 0,
                    'detections': 0,
                    'ground_truths': 0,
                    'false_positives': 0,
                    'false_negatives': 0
                }
            per_class_data[class_name]['ground_truths'] += 1

        detected_distances, ground_truth_distances, distance_errors, matches, unmatched_detections, unmatched_ground_truths = [], [], [], [], [], []
        if detection_list and ground_truth_list:
            matches, unmatched_detections, unmatched_ground_truths = match_detections_to_ground_truth_centroid(detection_list, ground_truth_list, distance_threshold=2.0)
            
            TP = len(matches)
            FP = len(unmatched_detections)
            FN = len(unmatched_ground_truths)
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0

            print_separator("Evaluation Metrics")
            print(f"True Positives (TP): {TP}")
            print(f"False Positives (FP): {FP}")
            print(f"False Negatives (FN): {FN}")
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")

            # Compare distances using matched pairs
            for match in matches:
                detection_idx, gt_idx, _ = match
                detection = detection_list[detection_idx]
                gt = ground_truth_list[gt_idx]
                class_name = gt['class']

                detected_distance = detection['distance']
                gt_distance = calculate_distance_to_convex_hull(gt['points'])
                compare_distances(detected_distance, gt_distance, gt['class'])
                detected_distances.append(detected_distance)
                ground_truth_distances.append(gt_distance)
                distance_error = abs(detected_distance - gt_distance)
                distance_errors.append(distance_error)

                per_class_data[class_name]['detected_distances'].append(detected_distance)
                per_class_data[class_name]['ground_truth_distances'].append(gt_distance)
                per_class_data[class_name]['distance_errors'].append(distance_error)
                per_class_data[class_name]['matches'] += 1

                print_separator("IoU Computation")
                iou = compute_iou_3d(detection['corners'], gt['corners'])
                print(f"\nIoU Results for Detection {j + 1}:")
                if iou != 0:
                    print(f" - Maximum IoU: {iou:.4f}")
                else:
                    print(" - No matching ground truth object found.")

            for detection_idx in unmatched_detections:
                detection = detection_list[detection_idx]
                class_name = detection['class']
                if class_name not in per_class_data:
                    per_class_data[class_name] = {
                        'detected_distances': [],
                        'ground_truth_distances': [],
                        'distance_errors': [],
                        'matches': 0,
                        'detections': 0,
                        'ground_truths': 0,
                        'false_positives': 0,
                        'false_negatives': 0
                    }
                per_class_data[class_name]['false_positives'] += 1

            for gt_idx in unmatched_ground_truths:
                gt = ground_truth_list[gt_idx]
                class_name = gt['class']
                if class_name not in per_class_data:
                    per_class_data[class_name] = {
                        'detected_distances': [],
                        'ground_truth_distances': [],
                        'distance_errors': [],
                        'matches': 0,
                        'detections': 0,
                        'ground_truths': 0,
                        'false_positives': 0,
                        'false_negatives': 0
                    }
                per_class_data[class_name]['false_negatives'] += 1
            avg_distance_error = np.mean(distance_errors) if distance_errors else 0
            print(f"\nAverage Distance Error: {avg_distance_error:.2f} m")

        del point_cloud, image
        gc.collect()

    if mse_compute:
        return per_class_data
    else:
        return processed_point_cloud, f_points, f_labels, calib, dt_rys, dt_ry_cams, labels

def compute_box_3d(dimensions, location, rotation_y):
    
    h, w, l = dimensions
    
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    R = np.array([[np.cos(rotation_y), 0, np.sin(rotation_y)],
                  [0, 1, 0],
                  [-np.sin(rotation_y), 0, np.cos(rotation_y)]])

    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d += np.array(location).reshape(3, 1)

    return corners_3d.T

def get_3d_box_corners(center: np.ndarray, size: tuple) -> np.ndarray:
    l, w, h = size
    x, y, z = center

    corners = np.array([
        [x - l / 2, y - w / 2, z - h / 2],
        [x - l / 2, y - w / 2, z + h / 2],
        [x - l / 2, y + w / 2, z - h / 2],
        [x - l / 2, y + w / 2, z + h / 2],
        [x + l / 2, y - w / 2, z - h / 2],
        [x + l / 2, y - w / 2, z + h / 2],
        [x + l / 2, y + w / 2, z - h / 2],
        [x + l / 2, y + w / 2, z + h / 2]
    ])
    
    return corners

def get_3d_box_corner_lidar(bbox_center, bbox_size):
    l, w, h = bbox_size
    x, y, z = bbox_center

    corners = np.array([
        [x - l / 2, y - w / 2, z - h / 2],
        [x - l / 2, y - w / 2, z + h / 2],
        [x - l / 2, y + w / 2, z - h / 2],
        [x - l / 2, y + w / 2, z + h / 2],
        [x + l / 2, y - w / 2, z - h / 2],
        [x + l / 2, y - w / 2, z + h / 2],
        [x + l / 2, y + w / 2, z - h / 2],
        [x + l / 2, y + w / 2, z + h / 2]
    ])

    return corners

def draw_3d_bbox_mayavi(corners_3d, fig, color=(1, 0, 0), is_gt=False):
    if is_gt:
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
    else:
        edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),
            (4, 5), (5, 7), (7, 6), (6, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
    ]
        
    for edge in edges:
        p1 = corners_3d[edge[0]]
        p2 = corners_3d[edge[1]]
        mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, tube_radius=None, line_width=1, figure=fig)

def transform_corners_to_lidar(corners_3d_cam, calib):
    R0_rect = calib['R0_rect']
    Tr_velo_to_cam = calib['Tr_velo_to_cam']

    R0_rect_4x4 = np.eye(4)
    R0_rect_4x4[:3, :3] = R0_rect

    Tr_velo_to_cam_4x4 = np.eye(4)
    Tr_velo_to_cam_4x4[:3, :4] = Tr_velo_to_cam

    velo_to_cam = R0_rect_4x4 @ Tr_velo_to_cam_4x4
    cam_to_velo = np.linalg.inv(velo_to_cam)
    corners_3d_cam_hom = np.hstack((corners_3d_cam, np.ones((corners_3d_cam.shape[0], 1))))

    corners_lidar_hom = (cam_to_velo @ corners_3d_cam_hom.T).T

    return corners_lidar_hom[:, :3]

def plot_lidar_data_with_frustums(pc, frustum_points_list, frustum_labels_list, fig1=None, labels=None, calib=None, bgcolor=(0, 0, 0), pts_scale=50):
    if fig1 is None:
        fig1 = mlab.figure(bgcolor=bgcolor, size=(1600, 1024))
    else:
        mlab.clf(fig1)

    def generate_random_color():
        return (random.random(), random.random(), random.random())

    label_color_map = {}

    mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color=(0.5, 0.5, 0.5), mode='point', scale_factor=pts_scale, figure=fig1)

    for i, (points, label) in enumerate(zip(frustum_points_list, frustum_labels_list)):
        mapped_label = map_yolo_to_kitti(label)

        if (mapped_label, i) not in label_color_map:
            label_color_map[(mapped_label, i)] = generate_random_color()

        color = label_color_map[(mapped_label, i)]
        mlab.points3d(points[:, 0], points[:, 1], points[:, 2], color=color, mode='sphere', scale_factor=0.05, figure=fig1)
        
        centroid = np.mean(points, axis=0)
        mlab.text3d(centroid[0], centroid[1] - 1, centroid[2] - 1, f'ID: {i} - {mapped_label}', scale=0.3, color=color, figure=fig1)

        bbox_size = calculate_bbox_size(points)
        bbox_center = calculate_centroid(points)
        corners = get_3d_box_corner_lidar(bbox_center, bbox_size)
        draw_3d_bbox_mayavi(corners, fig1, color=color)
        
    if labels and calib:
        for i, (label) in enumerate(labels):
            if label['type'] == 'DontCare':
                continue
            dimensions = label['dimensions']
            location = label['location']
            rotation_y = label['rotation_y']
            corners_3d_cam = compute_box_3d(dimensions, location, rotation_y)
            corners_3d_lidar = transform_corners_to_lidar(corners_3d_cam, calib)
            draw_3d_bbox_mayavi(corners_3d_lidar, fig1, color=(0, 1, 0), is_gt=True)
            centroid = np.mean(corners_3d_lidar, axis=0)
            label = label['type']
            mlab.text3d(centroid[0], centroid[1] + 1, centroid[2] + 1, f'GT ID: {i} - {label}', scale=0.2, color=(0, 1, 0), figure=fig1)

    mlab.view(azimuth=180, elevation=70, focalpoint=[0, 0, 20], distance=150.0, figure=fig1)
    mlab.show()

def get_filenames(split, num_images=100):
    image_dir = config['data_paths']['image_dir'].format(split=split)
    filenames = sorted([f[:-4] for f in os.listdir(image_dir) if f.endswith('.png')])[:num_images]
    return filenames

def plot_overall_distance_comparison(overall_per_class_data, split):
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    total_detected_distances = []
    total_ground_truth_distances = []
    total_distance_errors = []
    for class_name, data in overall_per_class_data.items():
        distance_errors = data['distance_errors']
        if len(distance_errors) == 0:
            continue
        total_detected_distances.extend(data['detected_distances'])
        total_ground_truth_distances.extend(data['ground_truth_distances'])
        total_distance_errors.extend(data['distance_errors'])

    axs[0, 0].scatter(range(len(total_detected_distances)), total_ground_truth_distances, c='g', label='Ground Truth Distances')
    axs[0, 0].scatter(range(len(total_detected_distances)), total_detected_distances, c='b', label='Detected Distances')
    axs[0, 0].set_xlabel('Detection Index')
    axs[0, 0].set_ylabel('Distance (meters)')
    axs[0, 0].set_title('Detected vs. Ground Truth Distances')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].hist(total_distance_errors, bins=30, color='purple', edgecolor='black')
    axs[0, 1].set_xlabel('Distance Error (meters)')
    axs[0, 1].set_ylabel('No. Matched Detections')
    axs[0, 1].set_title('Histogram of Distance Errors')
    axs[0, 1].grid(True)

    slope, intercept, r_value, p_value, std_err = linregress(total_ground_truth_distances, total_distance_errors)
    line = slope * np.array(total_ground_truth_distances) + intercept
    axs[1, 0].scatter(total_ground_truth_distances, total_distance_errors, c='b', marker='o', label='Data Points')
    axs[1, 0].plot(total_ground_truth_distances, line, 'r', label=f'Fit Line (R² = {r_value**2:.2f})')
    axs[1, 0].set_xlabel('Ground Truth Distance (meters)')
    axs[1, 0].set_ylabel('Absolute Distance Error (meters)')
    axs[1, 0].set_title('Error vs. Ground Truth Distance')
    axs[1, 0].grid(True)

    sorted_errors = np.sort(total_distance_errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    axs[1, 1].plot(sorted_errors, cdf, marker='.', linestyle='none')
    axs[1, 1].set_xlabel('Absolute Distance Error (meters)')
    axs[1, 1].set_ylabel('Cumulative Probability')
    axs[1, 1].set_title('CDF of Distance Errors')
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(f'dataset/kitti/{split}/segmented_images/combined_error_plots_full.png')
    plt.close()

def plot_distance_error_histograms(overall_per_class_data, split):
    for class_name, data in overall_per_class_data.items():
        distance_errors = data['distance_errors']
        if len(distance_errors) == 0:
            continue
        plt.figure()
        plt.hist(distance_errors, bins=30, color='blue', edgecolor='black')
        plt.xlabel('Distance Error (meters)')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Distance Errors for Class: {class_name}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'dataset/kitti/{split}/segmented_images/{class_name}_full_distance_error_histogram.png')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Frustum PointNet Inference')
    parser.add_argument('--filename', type=str, required=True, help='Filename without extension')
    parser.add_argument('--split', type=str, default='training', help='Dataset split: training or testing')
    parser.add_argument('--num_images', type=int, default=1, help='Number of images to process')
    args = parser.parse_args()
    filename = args.filename
    split = args.split
    num_images = args.num_images

    image_dir = config['data_paths']['image_dir'].format(split=split)
    point_cloud_dir = config['data_paths']['point_cloud_dir'].format(split=split)
    calib_dir = config['data_paths']['calib_dir'].format(split=split)
    label_dir = config['data_paths']['label_dir'].format(split=split) if split == 'training' else None
    model_path = config['model_paths']['frustum_pointnet']

    total_start_time = time.time()

    image_path = f'{image_dir}/{filename}.png'
    point_cloud_path = f'{point_cloud_dir}/{filename}.bin'
    calib_path = f'{calib_dir}/{filename}.txt'
    label_path = f'{label_dir}/{filename}.txt' if label_dir else None

    model = load_model(model_path)

    if num_images > 1:
        overall_per_class_data = {}
        filenames = get_filenames(split, num_images)
        print(f"num_images = {num_images}, type = {type(num_images)}")

        for filename in filenames:
            print(f"\nProcessing image: {filename}")
            image_path = f'{image_dir}/{filename}.png'
            point_cloud_path = f'{point_cloud_dir}/{filename}.bin'
            calib_path = f'{calib_dir}/{filename}.txt'
            label_path = f'{label_dir}/{filename}.txt' if label_dir else None

            per_class_data = run_inference(image_path, point_cloud_path, calib_path, label_path, model, filename, split, total_start_time, mse_compute=True)

            try:
                for class_name, data in per_class_data.items():
                    if class_name not in overall_per_class_data:
                        overall_per_class_data[class_name] = {
                            'detected_distances': [],
                            'ground_truth_distances': [],
                            'distance_errors': [],
                            'matches': 0,
                            'detections': 0,
                            'ground_truths': 0,
                            'false_positives': 0,
                            'false_negatives': 0
                        }
                    overall_per_class_data[class_name]['detected_distances'].extend(data['detected_distances'])
                    overall_per_class_data[class_name]['ground_truth_distances'].extend(data['ground_truth_distances'])
                    overall_per_class_data[class_name]['distance_errors'].extend(data['distance_errors'])
                    overall_per_class_data[class_name]['matches'] += data['matches']
                    overall_per_class_data[class_name]['detections'] += data['detections']
                    overall_per_class_data[class_name]['ground_truths'] += data['ground_truths']
                    overall_per_class_data[class_name]['false_positives'] += data['false_positives']
                    overall_per_class_data[class_name]['false_negatives'] += data['false_negatives']
            except Exception as e:
                print(f"Error processing image: {e}")
                continue
            
        print("\nPer-Class Evaluation Metrics:")
        for class_name, data in overall_per_class_data.items():
            detected_distances = np.array(data['detected_distances'])
            ground_truth_distances = np.array(data['ground_truth_distances'])
            distance_errors = np.array(data['distance_errors'])
            matches = data['matches']
            detections = data['detections']
            ground_truths = data['ground_truths']
            false_positives = data['false_positives']
            false_negatives = data['false_negatives']
            precision = matches / detections if detections > 0 else 0
            recall = matches / ground_truths if ground_truths > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            mean_distance_error = np.mean(distance_errors) if len(distance_errors) > 0 else 0
            mse = np.mean(distance_errors ** 2) if len(distance_errors) > 0 else 0
            print(f"\nClass: {class_name}")
            print(f" - Matches: {matches}")
            print(f" - Detections: {detections}")
            print(f" - Ground Truths: {ground_truths}")
            print(f" - False Positives: {false_positives}")
            print(f" - False Negatives: {false_negatives}")
            print(f" - Precision: {precision:.2f}")
            print(f" - Recall: {recall:.2f}")
            print(f" - F1 Score: {f1_score:.2f}")
            print(f" - Mean Distance Error: {mean_distance_error:.2f} m")
            print(f" - MSE of Distance Errors: {mse:.4f} m^2")

            plot_distance_error_histograms(overall_per_class_data, split)
            plot_overall_distance_comparison(overall_per_class_data, split)   

    else:
        processed_point_cloud, f_points, f_labels, calib, dt_rys, dt_ry_cams, gt_labels = run_inference(image_path, point_cloud_path, calib_path, label_path, model, filename, split, total_start_time, mse_compute=False)
        
        print("-" * 50)
        total_end_time = time.time()
        print(f"Total execution time: {total_end_time - total_start_time:.3f}s.")
        print("-" * 50)
        segmented_image = cv2.imread(f'dataset/kitti/{split}/segmented_images/{filename}_3D_box.png')
        cv2.imshow("Segmented Image", segmented_image)
        if label_path:
            ground_truth_image = cv2.imread(f'dataset/kitti/{split}/segmented_images/{filename}_compare.png')
            cv2.imshow("Ground truth Image", ground_truth_image)
        plot_lidar_data_with_frustums(processed_point_cloud, f_points, f_labels, labels=gt_labels, calib=calib)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
