import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import os
import numpy as np
import zipfile
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from CIFAR import train_loader, custom_CIFAR10

# model training
model_res = models.resnet18()

# Simple Learning Rate Scheduler
def lr_scheduler(optimizer, epoch):
    lr = learning_rate
    if epoch >= 50:
        lr /= 10
    if epoch >= 100:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Xavier
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

model_res.apply(init_weights)
#device='cuda'
# 서버를 사용한다면 번호를 지정
#model_res = model_res.to(device)

learning_rate = 0.1
num_epoch = 50
model_name = 'res_model.pth'

# loss 함수
loss_fn = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.SGD(model_res.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)

train_loss = 0
valid_loss = 0
correct = 0
total_cnt = 0
best_acc = 0

# Train
for epoch in range(num_epoch):
    print(f"====== { epoch+1} epoch of { num_epoch } ======")
    model_res.train()
    lr_scheduler(optimizer, epoch)
    train_loss = 0
    valid_loss = 0
    correct = 0
    total_cnt = 0
    for step, batch in enumerate(train_loader):
        #  input and target
        batch[0], batch[1] = batch[0], batch[1]
        optimizer.zero_grad()

        logits = model_res(batch[0])
        loss = loss_fn(logits, batch[1])
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predict = logits.max(1)

        total_cnt += batch[1].size(0)
        correct +=  predict.eq(batch[1]).sum().item()

        if step % 100 == 0 and step != 0:
            print(f"\n====== { step } Step of { len(train_loader) } ======")
            print(f"Train Acc : { correct / total_cnt }")
            print(f"Train Loss : { loss.item() / batch[1].size(0) }")

