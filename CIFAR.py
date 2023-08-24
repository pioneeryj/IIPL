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

############################
## Custom Dataset ##########
############################

file_path = 'C:/Users/pione/Desktop/IIPL/CIFAR/cifar10/cifar10/train'

# os.walk 출력 (경로, 경로 내 디렉토리 리스트, 경로 내 파일 리스트)
for (path, dir, file) in os.walk(file_path):
    #print(path)
    print(dir) #class
    #print(file) #png img


#class_name, class_num mapping
class_name=sorted(os.listdir(path))
class_num=list(range(len(class_name)))
class_mapping=dict(zip(class_name,class_num))


file_path_lst=[]
class_num_lst=[]
for (path, dir, file) in os.walk(file_path):  
       for filename in file:
          file_path_lst=file_path_lst.append(os.path.join(dir,filename))
          class_num_lst=class_num_lst.append(class_mapping[dir])
        
    
       
len(file_path_lst)
len(class_num_lst)  
path_class_lst=zip(file_path_lst,class_num_lst)


# making custom dataset
class custom_CIFAR10():
  def __init__(self,path_class_lst,train=True):
    transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(256),
    transforms.CenterCrop(224),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    # resnet 모델 기준에 맞춰서 transform

    self.path_class_lst=path_class_lst
    

  def __len__(self):
    return len(self.path_class_lst)


  def __getitem__(self, idx):
    path, label = self.path_class_lst[idx]
    img=Image.open(path).convert('RGB')
    return img, label 
  
  #img와 label을 return

#=========================================================================================================
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

loss_fn = nn.CrossEntropyLoss()
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
    # Train Phase
    for step, batch in enumerate(train_loader):
        #  input and target
        batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
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

    correct = 0
    total_cnt = 0


# train 끝나고 모델 저장


# Test Phase
    with torch.no_grad():
        model_res.eval()
        for step, batch in enumerate(test_loader):
            # input and target
            batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
            total_cnt += batch[1].size(0)
            logits = model_res(batch[0])
            valid_loss += loss_fn(logits, batch[1])
            _, predict = logits.max(1)
            correct += predict.eq(batch[1]).sum().item()
        valid_acc = correct / total_cnt
        print(f"\nValid Acc : { valid_acc }")
        print(f"Valid Loss : { valid_loss / total_cnt }")

        if(valid_acc > best_acc):
            best_acc = valid_acc
            torch.save(model_res, model_name)
            print("Model Saved!")