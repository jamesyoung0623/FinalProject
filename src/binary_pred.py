import os
import model
import data
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import csv
import data
from model import model
from model import binary_classification_model
from PIL import Image

test_set = data.data('test', [])
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

model_A = binary_classification_model()
model_A.load_state_dict(torch.load(os.path.join('model/modelA.tar')))
model_A.eval()

model_B = binary_classification_model()
model_B.load_state_dict(torch.load(os.path.join('model/modelB.tar')))
model_B.eval()

model_C = binary_classification_model()
model_C.load_state_dict(torch.load(os.path.join('model/modelC.tar')))
model_C.eval()

file = open('predict.csv', 'w')
file.write('image_id,label\n')

with torch.no_grad():
    for idx, (name, image) in enumerate(test_loader):
        print('evaluating image {}'.format(idx))
        pred_A = model_A(image)[0]
        pred_B = model_B(image)[0]
        pred_C = model_C(image)[0]
        pred = max(pred_A, pred_B, pred_C)
        
        if pred == pred_A:
            file.write('{},{}\n'.format(name[0], 'A'))
        elif pred == pred_B:
            file.write('{},{}\n'.format(name[0], 'B'))
        elif pred == pred_C:
            file.write('{},{}\n'.format(name[0], 'C'))
