import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.resnet = models.resnext101_32x8d(pretrained=True)
        self.fc1 = nn.Linear(1000, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 3)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.resnet(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x
    
class binary_classification_model(nn.Module):
    def __init__(self):
        super(binary_classification_model, self).__init__()
        self.resnet = models.resnext101_32x8d(pretrained=True)
        self.fc1 = nn.Linear(1000, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 4)
        self.fc5 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.resnet(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = self.sigmoid(x)
        return x