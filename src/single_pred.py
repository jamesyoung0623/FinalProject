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

val_label = []

val_label_file = open('data/dev.csv', 'r')
for row in val_label_file:
    if row[10] == 'A':
        val_label.append([1, 0, 0])
    elif row[10] == 'B':
        val_label.append([0, 1, 0])
    elif row[10] == 'C':
        val_label.append([0, 0, 1])
    
        
val_set = data.data('val', val_label)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False)

test_set = data.data('test', [])
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

file = open('predict.csv', 'w')
file.write('image_id,label\n')

model = model()
model.load_state_dict(torch.load(os.path.join('model/model.tar')))
model.eval()

pred_list = []
label_list = []

for idx, (image, label) in enumerate(val_loader):
    print('evaluating image {}'.format(idx))
    output = model(image)
    pred = torch.max(output[0])
    for i in range(3):
        if output[0][i] == pred:
            if i == 0:
                pred_list.append('A')
            elif i == 1:
                pred_list.append('B')
            else:
                pred_list.append('C')
    for i in range(3):
        if label[0][i] == 1.0:
            if i == 0:
                label_list.append('A')
            elif i == 1:
                label_list.append('B')
            else:
                label_list.append('C')

acc = 0.0
for i in range(len(val_label)):
    if pred_list[i] == label_list[i]:
        acc += 1
print('ACC: {}'.format(acc/len(val_label)))

for idx, (name, image) in enumerate(test_loader):
    print('evaluating image {}'.format(idx))
    output = model(image)
    pred = torch.max(output[0])
    for i in range(3):
        if output[0][i] == pred:
            if i == 0:
                file.write('{},{}\n'.format(name[0], 'A'))
            elif i == 1:
                file.write('{},{}\n'.format(name[0], 'B'))
            else:
                file.write('{},{}\n'.format(name[0], 'C'))

