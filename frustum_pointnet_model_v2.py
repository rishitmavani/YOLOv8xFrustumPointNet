import torch
import torch.nn as nn
import torch.nn.functional as F

class FrustumPointNet(nn.Module):
    def __init__(self, num_classes=9):
        super(FrustumPointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)        
        self.fc1 = nn.Linear(1024, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(512, 256)
        self.fc_bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.fc_bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)
        
        return x

#model = FrustumPointNet(num_classes=9) 
#model.summary()
# print(model)
