import os
import cv2
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import data
from model import model

BATCH = 20
EPOCHS = 50

train_label = []
val_label = []

train_label_file = open('data/train.csv', 'r')
for row in train_label_file:
    if row[10] == 'A':
        train_label.append([1, 0, 0])
    elif row[10] == 'B':
        train_label.append([0, 1, 0])
    elif row[10] == 'C':
        train_label.append([0, 0, 1])

val_label_file = open('data/dev.csv', 'r')
for row in val_label_file:
    if row[10] == 'A':
        val_label.append([1, 0, 0])
    elif row[10] == 'B':
        val_label.append([0, 1, 0])
    elif row[10] == 'C':
        val_label.append([0, 0, 1])
    
        
print('Loading data...')
train_set = data.data('train', train_label)
val_set = data.data('val', val_label)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_set, batch_size=BATCH, shuffle=False)

model = model()
model.cuda()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
loss_function = nn.MSELoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

best_acc = 0.0
print('Start training...')

for epoch in range(EPOCHS):
    train_acc = 0.0
    val_acc = 0.0
    
    model.train()
    for idx, (image, label) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(image.cuda())
        loss = loss_function(output, label.cuda())
        loss.backward()
        optimizer.step()
        print('Epoch: [{0}][{1}/{2}] loss: {3}'.format(epoch+1, idx+1, len(train_loader), loss.item()))
    
    model.eval()
    with torch.no_grad():
        for idx, (image, label) in enumerate(val_loader):
            output = model(image.cuda())
            for i in range(BATCH):
                pred = torch.max(output[i])
                for j in range(3):
                    if output[i][j] == pred and label[i][j] == 1.0:
                        val_acc += 1
                        break
        
    print('Epoch: [{0}] val_acc: {1}'.format(epoch + 1, val_acc/len(val_label)))
    if val_acc/len(val_label) > best_acc:
        best_acc = val_acc/len(val_label)
        torch.save(model.state_dict(), os.path.join('model/model.tar'))
    scheduler.step()
