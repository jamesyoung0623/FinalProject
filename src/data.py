import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class data(Dataset):
    def __init__(self, data_type, labels):
        self.data_type = data_type
        self.images = []
        self.labels = torch.FloatTensor(labels)
        
        if(self.data_type == 'train'):
            self.dir = os.path.join('../data/train/')
            self.image_names = sorted(os.listdir(self.dir)) 
            
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)), 
                transforms.RandomRotation(15), 
                transforms.RandomHorizontalFlip(), 
                transforms.ColorJitter(),
                transforms.ToTensor(), 
                transforms.Normalize(MEAN, STD) 
            ])
                                        
        elif(self.data_type == 'val'):
            self.dir = os.path.join('../data/dev/')
            self.image_names = sorted(os.listdir(self.dir)) 
            
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)), 
                transforms.ColorJitter(),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD)
            ])
    
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.dir+self.image_names[idx]).convert('RGB'))
        return image, self.labels[idx]