import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import pad
from frustum_pointnet_model_v2 import FrustumPointNet

YOLO_TO_KITTI = {
    0.0: 0,  # person to Pedestrian
    1.0: 1,  # bicycle to Bicycle
    2.0: 2,  # car to Car
    3.0: 3,  # motorcycle to Motorcycle
    4.0: 6,  # airplane to Misc
    5.0: 5,  # bus to Van
    6.0: 4,  # train to Train
    7.0: 7,   # truck to Truck
    8.0: 6,  # boat to Misc
}

def map_yolo_to_kitti(yolo_class_id):
    return YOLO_TO_KITTI.get(yolo_class_id, 6)

class FrustumDataset(Dataset):
    def __init__(self, frustum_points, frustum_labels, bounding_boxes, masks):
        self.frustum_points = frustum_points
        self.frustum_labels = [map_yolo_to_kitti(label) for label in frustum_labels]
        self.bounding_boxes = bounding_boxes
        self.masks = masks
    
    def __len__(self):
        return len(self.frustum_points)
    
    def __getitem__(self, idx):
        points = torch.tensor(self.frustum_points[idx][:, :3], dtype=torch.float32).transpose(0, 1)
        label = torch.tensor(self.frustum_labels[idx], dtype=torch.long)
        bounding_boxes = torch.tensor(self.bounding_boxes[idx], dtype=torch.float32)
        mask = torch.tensor(self.masks[idx], dtype=torch.float32)
        return points, label, bounding_boxes, mask

def pad_collate(batch):
    (points, labels, bounding_boxes, masks) = zip(*batch)

    max_len = max(p.size(1) for p in points)
    padded_points = []
    for points_i in points:
        pad_size = max_len - points_i.size(1)
        padded_points.append(torch.nn.functional.pad(points_i, (0, pad_size), "constant", 0))
    
    padded_points = torch.stack(padded_points)

    labels_long = []
    for labels_i in labels:
        labels_long.append(labels_i.long())

    labels_stack = torch.stack(labels_long)
    bounding_boxes_stack = torch.stack(bounding_boxes)

    padded_masks = [torch.nn.functional.pad(m, (0, max_len - m.size(0)), "constant", 0) for m in masks]
    masks_stack = torch.stack(padded_masks)

    return padded_points, labels_stack, bounding_boxes_stack, masks_stack

frustum_proposals = np.load('frustum_data_all_kitti_data.npz', allow_pickle=True)
frustum_points = [frustum_proposals[f'points_{i}'] for i in range(len(frustum_proposals) // 4)]
frustum_labels = [frustum_proposals[f'labels_{i}'] for i in range(len(frustum_proposals) // 4)]
bounding_boxes = [frustum_proposals[f'bounding_boxes_{i}'] for i in range(len(frustum_proposals) // 4)]
masks = [frustum_proposals[f'masks_{i}'] for i in range(len(frustum_proposals) // 4)]
frustum_labels = np.array(frustum_labels)

unique_labels = np.unique(frustum_labels)
mapped_labels = np.clip([map_yolo_to_kitti(label) for label in unique_labels], 0, 8)
image_width = 640
image_height = 224
bounding_boxes = [(max(0, min(x1, image_width)), max(0, min(y1, image_height)), max(0, min(x2, image_width)), max(0, min(y2, image_height))) for x1, y1, x2, y2 in bounding_boxes]

masks = [mask & (mask >= 0) & (mask < len(frustum_points[i])) for i, mask in enumerate(masks)]

def train():
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        print('Training...')
        model.train()
        running_loss = 0.0
        for points, labels, bounding_boxes, masks in dataloader:
            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}')
        print(f'Finished Training Epoch {epoch + 1}/{num_epochs}')

dataset = FrustumDataset(frustum_points, frustum_labels, bounding_boxes, masks)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=pad_collate)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
if device == 'cpu':
    print("Using CPU for inference.")
else:
    print("Using GPU for inference.")
model = FrustumPointNet(num_classes=9)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if __name__ == '__main__':
    train()
    torch.save(model.state_dict(), 'frustum_pointnet.pth')
