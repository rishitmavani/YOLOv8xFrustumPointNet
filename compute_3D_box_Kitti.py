import numpy as np
import cv2

def read_calibration(calib_path):
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

def load_ground_truth(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    objects = []
    for line in lines:
        parts = line.strip().split()
        obj = {
            'type': parts[0],
            'truncated': float(parts[1]),
            'occluded': int(parts[2]),
            'alpha': float(parts[3]),
            'bbox': [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])],
            'h': float(parts[8]),
            'w': float(parts[9]),
            'l': float(parts[10]),
            'x': float(parts[11]),
            'y': float(parts[12]),
            'z': float(parts[13]),
            'dimensions': [float(parts[8]), float(parts[9]), float(parts[10])],
            'location': [float(parts[11]), float(parts[12]), float(parts[13])],
            'rotation_y': float(parts[14])
        }
        if obj['type'] == 'DontCare':
            continue
        objects.append(obj)
    return objects

def compute_box_3d(dim, loc, ry):
    R = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    h, w, l = dim

    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]  # Adjusted to use ground plane
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))

    corners_3d[0, :] += loc[0]
    corners_3d[1, :] += loc[1]
    corners_3d[2, :] += loc[2]

    return corners_3d

def project_to_image_kitti(pts_3d, P, R0_rect):
    if len(pts_3d.shape) == 1:
        pts_3d = np.expand_dims(pts_3d, axis=0)
    
    pts_3d_rect = R0_rect @ pts_3d[:3, :]
    
    pts_2d_cam = P @ np.vstack((pts_3d_rect, np.ones((1, pts_3d_rect.shape[1]))))
    pts_2d_cam = pts_2d_cam[:2, :] / pts_2d_cam[2, :]
    
    return pts_2d_cam.T

def draw_3d_box(image, corners_2d, obj_type, box, obj_id, color=(0, 255, 0)):
    x1, y1, x2, y2 = box
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    for edge in edges:
        pt1 = tuple(corners_2d[edge[0]].astype(int))
        pt2 = tuple(corners_2d[edge[1]].astype(int))
        cv2.line(image, pt1, pt2, color, 2)
    
    label = f"{obj_type} {obj_id}"

    cv2.putText(image, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return image

def draw_boxes_on_image(image, calib_path, label_path):
    calib = read_calibration(calib_path)
    
    ground_truth_objects = load_ground_truth(label_path)

    object_ids, dimensions, locations, types, rotation_y = [], [], [], [], []

    i = 0
    for obj in ground_truth_objects:
        if obj['type'] == 'DontCare':
            continue

        corners_3d = compute_box_3d(obj['dimensions'], obj['location'], obj['rotation_y'])
        #print("3D Bounding Box Corners (World Coordinates):", corners_3d)
        #print(f"GT Location: {obj['location']}, GT Rotation: {obj['rotation_y']}, GT Dimensions: {obj['dimensions']}, GT Type: {obj['type']}")
        corners_2d = project_to_image_kitti(corners_3d, calib['P2'], calib['R0_rect'])
        #print("Corners 2D (after projection):", corners_2d)

        image = draw_3d_box(image, np.array(corners_2d), obj['type'], obj['bbox'], i)

        dimensions.append([obj['l'], obj['w'], obj['h']])
        locations.append([obj['z'], obj['y'], obj['x']])
        types.append(obj['type'])
        rotation_y.append(obj['rotation_y'])
        object_ids.append(i)
        i += 1
    return image
    # cv2.imshow("Image with 3D Boxes", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# if __name__ == "__main__":
#     split = 'training'
#     filename = '000019'
#     image_path = f'dataset/kitti/{split}/image_2/{filename}.png'
#     point_cloud_path = f'dataset/kitti/{split}/velodyne/{filename}.bin'
#     calib_path = f'dataset/kitti/{split}/calib/{filename}.txt'
#     model_path = f'frustum_pointnet_v4.pth'
#     label_path = f'dataset/kitti/{split}/label_2/{filename}.txt'
    
#    draw_boxes_on_image(image_path, calib_path, label_path)
