import os
import cv2
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import data
from model import binary_classification_model

BATCH = 20
EPOCHS = 50

train_label_A = []
train_label_B = []
train_label_C = []
val_label_A = []
val_label_B = []
val_label_C = []
val_label = []

train_label_file = open('../data/train.csv', 'r')
for row in train_label_file:
    if row[10] == 'A':
        train_label_A.append(1)
        train_label_B.append(0)
        train_label_C.append(0)
    elif row[10] == 'B':
        train_label_A.append(0)
        train_label_B.append(1)
        train_label_C.append(0)
    elif row[10] == 'C':
        train_label_A.append(0)
        train_label_B.append(0)
        train_label_C.append(1)

val_label_file = open('../data/dev.csv', 'r')
for row in val_label_file:
    if row[10] == 'A':
        val_label_A.append(1)
        val_label_B.append(0)
        val_label_C.append(0)
        val_label.append([1, 0, 0])
    elif row[10] == 'B':
        val_label_A.append(0)
        val_label_B.append(1)
        val_label_C.append(0)
        val_label.append([0, 1, 0])
    elif row[10] == 'C':
        val_label_A.append(0)
        val_label_B.append(0)
        val_label_C.append(1)
        val_label.append([0, 0, 1])
        
print('Loading data...')
train_set_A = data.data('train', train_label_A)
train_set_B = data.data('train', train_label_B)
train_set_C = data.data('train', train_label_C)
val_set_A = data.data('val', val_label_A)
val_set_B = data.data('val', val_label_B)
val_set_C = data.data('val', val_label_C)
val_set = data.data('val', val_label)

train_loader_A = torch.utils.data.DataLoader(train_set_A, batch_size=BATCH, shuffle=True)
train_loader_B = torch.utils.data.DataLoader(train_set_B, batch_size=BATCH, shuffle=True)
train_loader_C = torch.utils.data.DataLoader(train_set_C, batch_size=BATCH, shuffle=True)
val_loader_A = torch.utils.data.DataLoader(val_set_A, batch_size=BATCH, shuffle=False)
val_loader_B = torch.utils.data.DataLoader(val_set_B, batch_size=BATCH, shuffle=False)
val_loader_C = torch.utils.data.DataLoader(val_set_C, batch_size=BATCH, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False)

model_A = binary_classification_model()
model_A = model_A.cuda()
optimizer = optim.SGD(model_A.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
loss_function = nn.MSELoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
best_acc = 0.0
print('Start training...')

for epoch in range(EPOCHS):
    train_acc = 0.0
    val_acc = 0.0
    
    model_A.train()
    for idx, (image, label) in enumerate(train_loader_A):
        optimizer.zero_grad()
        image = image.cuda()
        label = label.cuda()
        output = model_A(image)[:, 0]
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()
        image = image.cpu()
        label = label.cpu()
        print('Epoch: [{0}][{1}/{2}] loss: {3}'.format(epoch+1, idx+1, len(train_loader_A), loss.item()))
    
    model_A.eval()
    #with torch.no_grad():
    #    for idx, (image, label) in enumerate(train_loader_A):
    #        for i in range(BATCH):
    #            image = image.cuda()
    #            output = model_A(image)[i][0]
    #            pred = 1 if output > 0.5 else 0
    #            if pred == label[i]:
    #                train_acc += 1
    #            image = image.cpu()
        
    #print('Epoch: [{0}] train_acc_A: {1}'.format(epoch + 1, train_acc/len(train_label_A)))
    
    with torch.no_grad():
        for idx, (image, label) in enumerate(val_loader_A):
            for i in range(BATCH):
                image = image.cuda()
                output = model_A(image.cuda())[i][0]
                pred = 1 if output > 0.5 else 0
                if pred == label[i]:
                    val_acc += 1
                image = image.cpu()
        
    print('Epoch: [{0}] val_acc_A: {1}'.format(epoch + 1, val_acc/len(val_label_A)))
    if val_acc/len(val_label_A) > best_acc:
        best_acc = val_acc/len(val_label_A)
        torch.save(model_A.state_dict(), os.path.join('../model/modelA.tar'))
    scheduler.step()
model_A = model_A.cpu()
    
model_B = binary_classification_model()
model_B = model_B.cuda()
optimizer = optim.SGD(model_B.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
loss_function = nn.MSELoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
best_acc = 0.0

for epoch in range(EPOCHS):
    train_acc = 0.0
    val_acc = 0.0
    
    model_B.train()
    for idx, (image, label) in enumerate(train_loader_B):
        optimizer.zero_grad()
        image = image.cuda()
        label = label.cuda()
        output = model_B(image)[:, 0]
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()
        image = image.cpu()
        label = label.cpu()
        print('Epoch: [{0}][{1}/{2}] loss: {3}'.format(epoch+1, idx+1, len(train_loader_B), loss.item()))
    
    model_B.eval()
    #with torch.no_grad():
    #    for idx, (image, label) in enumerate(train_loader_B):
    #        for i in range(BATCH):
    #            image = image.cuda()
    #            output = model_B(image)[i][0]
    #            pred = 1 if output > 0.5 else 0
    #            if pred == label[i]:
    #                train_acc += 1
    #            image = image.cpu()
    #print('Epoch: [{0}] train_acc_B: {1}'.format(epoch + 1, train_acc/len(train_label_B)))
    
    with torch.no_grad():
        for idx, (image, label) in enumerate(val_loader_B):
            for i in range(BATCH):
                image = image.cuda()
                output = model_B(image)[i][0]
                pred = 1 if output > 0.5 else 0
                if pred == label[i]:
                    val_acc += 1
                image = image.cpu()
        
    print('Epoch: [{0}] val_acc_B: {1}'.format(epoch + 1, val_acc/len(val_label_B)))
    if val_acc/len(val_label_B) > best_acc:
        best_acc = val_acc/len(val_label_B)
        torch.save(model_B.state_dict(), os.path.join('../model/modelB.tar'))
    scheduler.step()
model_B = model_B.cpu()

model_C = binary_classification_model()
model_C = model_C.cuda()
optimizer = optim.SGD(model_C.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
loss_function = nn.MSELoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
best_acc = 0.0

for epoch in range(EPOCHS):
    train_acc = 0.0
    val_acc = 0.0
    
    model_C.train()
    for idx, (image, label) in enumerate(train_loader_C):
        optimizer.zero_grad()
        image = image.cuda()
        label = label.cuda()
        output = model_C(image)[:, 0]
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()
        image = image.cpu()
        label = label.cpu()
        print('Epoch: [{0}][{1}/{2}] loss: {3}'.format(epoch+1, idx+1, len(train_loader_C), loss.item()))
    
    model_C.eval()
    #with torch.no_grad():
    #    for idx, (image, label) in enumerate(train_loader_C):
    #        for i in range(BATCH):
    #            image = image.cuda()
    #            output = model_C(image)[i][0]
    #            pred = 1 if output > 0.5 else 0
    #            if pred == label[i]:
    #                train_acc += 1
    #            image = image.cpu()
        
    #print('Epoch: [{0}] train_acc_C: {1}'.format(epoch + 1, train_acc/len(train_label_C)))
    
    with torch.no_grad():
        for idx, (image, label) in enumerate(val_loader_C):
            for i in range(BATCH):
                image = image.cuda()
                output = model_C(image)[i][0]
                pred = 1 if output > 0.5 else 0
                if pred == label[i]:
                    val_acc += 1
                image = image.cpu()
        
    print('Epoch: [{0}] val_acc_C: {1}'.format(epoch + 1, val_acc/len(val_label_C)))
    if val_acc/len(val_label_C) > best_acc:
        best_acc = val_acc/len(val_label_C)
        torch.save(model_C.state_dict(), os.path.join('../model/modelC.tar'))
    scheduler.step()
model_C = model_C.cpu()

acc = 0.0
with torch.no_grad():
    for idx, (image, label) in enumerate(val_loader):
        pred_A = model_A(image)[0]
        pred_B = model_B(image)[0]
        pred_C = model_C(image)[0]
        pred = max(pred_A, pred_B, pred_C)
        if pred == pred_A and label[0][0] == 1.0:
            acc += 1
        elif pred == pred_B and label[0][1] == 1.0:
            acc += 1
        elif pred == pred_C and label[0][2] == 1.0:
            acc += 1

print('ACC: {}'.format(acc/len(val_label)))