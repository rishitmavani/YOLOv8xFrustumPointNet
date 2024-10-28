import os
import numpy as np
import cv2
from yolov8n_segmentation import YOLOv8nSegmentation

def read_calib_file(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()

    calib = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        key, value = line.split(':', 1)
        calib[key] = np.array([float(x) for x in value.split()])
    calib['P2'] = calib['P2'].reshape(3, 4)
    calib['R0_rect'] = calib['R0_rect'].reshape(3, 3)
    calib['Tr_velo_to_cam'] = calib['Tr_velo_to_cam'].reshape(3, 4)
    return calib

class KITTIDataset:
    def __init__(self, base_path, mode='training', train_dataset_no=2000):
        self.base_path = base_path
        self.mode = mode
        self.image_path = os.path.join(base_path, mode, 'image_2')
        self.velo_path = os.path.join(base_path, mode, 'velodyne')
        self.calib_path = os.path.join(base_path, mode, 'calib')
        self.label_path = os.path.join(base_path, mode, 'label_2')
        self.image_files = sorted(os.listdir(self.image_path))
        self.velo_files = sorted(os.listdir(self.velo_path))
        self.calib_files = sorted(os.listdir(self.calib_path))
        self.label_files = sorted(os.listdir(self.label_path))
    
    def get_image(self, index):
        image_file = os.path.join(self.image_path, self.image_files[index])
        image = cv2.imread(image_file)
        return image, self.image_files[index]
    
    def get_velo(self, index):
        velo_file = os.path.join(self.velo_path, self.velo_files[index])
        point_cloud = np.fromfile(velo_file, dtype=np.float32).reshape(-1, 4)
        return point_cloud
    
    def get_calib(self, index):
        calib_file = os.path.join(self.calib_path, self.calib_files[index])
        calib = read_calib_file(calib_file)
        return calib

    def get_label(self, index):
        label_file = os.path.join(self.label_path, self.label_files[index])
        with open(label_file, 'r') as f:
            lines = f.readlines()
        labels = []
        for line in lines:
            line = line.strip().split(' ')
            if line[0] in ['Car', 'Van', 'Truck', 'Pedestrian', 'Cyclist', 'Tram', 'Train', 'Misc']:
                label = {
                    'type': line[0],
                    'truncated': float(line[1]),
                    'occluded': int(line[2]),
                    'alpha': float(line[3]),
                    'bbox': [float(x) for x in line[4:8]],
                    'dimensions': [float(x) for x in line[8:11]],
                    'location': [float(x) for x in line[11:14]],
                    'rotation_y': float(line[14])
                }
                labels.append(label)
        return labels

def project_lidar_to_image(calib, points):
    if len(points.shape) == 1:
        points = np.expand_dims(points, axis=0)
    
    points = points[:, :3]

    pts_3d_velo = np.hstack((points, np.ones((points.shape[0], 1))))
    
    pts_3d_cam = calib['Tr_velo_to_cam'] @ pts_3d_velo.T
    pts_3d_cam = pts_3d_cam[:3, :]
    
    pts_3d_cam = calib['R0_rect'] @ pts_3d_cam
    
    pts_2d_cam = calib['P2'] @ np.vstack((pts_3d_cam, np.ones((1, pts_3d_cam.shape[1]))))
    pts_2d_cam = pts_2d_cam[:2, :] / pts_2d_cam[2, :]
    
    return pts_2d_cam.T

def generate_frustum_proposals(base_path, mode, model_path, train_dataset_no=2000):
    kitti_data = KITTIDataset(base_path, mode, train_dataset_no)
    segmentation_model = YOLOv8nSegmentation(model_path)
    
    frustum_points_list = []
    frustum_labels_list = []
    bounding_boxes_list = []
    masks_list = []
    for i in range(train_dataset_no):
        image, index = kitti_data.get_image(i)
        print(f"Processing frame {index}")
        point_cloud = kitti_data.get_velo(i)
        calib = kitti_data.get_calib(i)
        labels = kitti_data.get_label(i)
        
        results = segmentation_model.segment(image)
        
        for result in results:
            filename = os.path.join(base_path, mode, "segmented_images", index)
            print(f"Saving segmented image to {filename}")
            result.save(filename)
            segmented_image = cv2.imread(filename)
            if hasattr(result, 'boxes') and result.boxes is not None:
                bounding_boxes = result.boxes
                masks = result.masks
                boxes = bounding_boxes.xyxy.cpu().numpy()
                confidences = bounding_boxes.conf.cpu().numpy()
                classes = bounding_boxes.cls.cpu().numpy()
                
                for box, conf, cls in zip(boxes, confidences, classes):
                    x1, y1, x2, y2 = box
                    if conf < 0.45:
                        continue
                    pts_2d_cam = project_lidar_to_image(calib, point_cloud[:, :3])
                    
                    mask = (pts_2d_cam[:, 0] >= x1) & (pts_2d_cam[:, 0] <= x2) & \
                           (pts_2d_cam[:, 1] >= y1) & (pts_2d_cam[:, 1] <= y2)
                    
                    frustum_points = point_cloud[mask]
                    
                    print(f"Frame {i}: Bounding box {box}, Cls: {cls}, Conf: {conf}, Points in frustum: {len(frustum_points)}")
                    if len(frustum_points) > 0:
                        frustum_points_list.append(frustum_points)
                        frustum_labels_list.append(cls)
                        bounding_boxes_list.append((x1, y1, x2, y2))
                        masks_list.append(mask)
    
    return frustum_points_list, frustum_labels_list, bounding_boxes_list, masks_list, segmented_image, result

# base_path = 'E:/BachelorThesis/KITTI_Dataset'
# mode = 'training'
# model_path = 'yolov8n-seg.pt'
# train_dataset_no = 2000
# frustum_points, frustum_labels, bounding_boxes_list, masks_list, segmented_image, result = generate_frustum_proposals(base_path, mode, model_path, train_dataset_no)

# if len(frustum_points) == 0:
#     print("No frustum proposals generated.")
# else:
#     print(f"Generated {len(frustum_points)} frustum proposals.")

# np.savez('frustum_data_low_conf', **{f'points_{i}': frustum_points[i] for i in range(len(frustum_points))}, **{f'labels_{i}': frustum_labels[i] for i in range(len(frustum_labels))}, **{f'bounding_boxes_{i}': bounding_boxes_list[i] for i in range(len(bounding_boxes_list))}, **{f'masks_{i}': masks_list[i] for i in range(len(masks_list))})
