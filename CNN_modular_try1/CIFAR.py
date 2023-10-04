import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import os
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
#import matplotlib.pyplot as plt

############################
##     Custom Dataset     ##
############################

'''file_path = 'C:/Users/pione/Desktop/IIPL/CIFAR/cifar10/cifar10/train'

# os.walk 출력 (경로, 경로 내 디렉토리 리스트, 경로 내 파일 리스트)
#for (path, dir, file) in os.walk(file_path):
    #print(path)


#class_name, class_num mapping

class_name=sorted(os.listdir(file_path))
class_num=list(range(len(class_name)))
class_mapping=dict(zip(class_name,class_num))
print(class_name)


file_path_lst=[]
class_num_lst=[]

# "train" 폴더 안의 모든 클래스(폴더)를 순회
for class_name in os.listdir(file_path):
    class_dir = os.path.join(file_path, class_name)
    
    # 각 클래스(폴더) 내의 이미지 파일들을 순회
    for filename in os.listdir(class_dir):
        image_path = os.path.join(class_dir, filename)
        
        # 파일 경로와 해당 이미지의 클래스(폴더)를 리스트에 추가
        file_path_lst.append(image_path)
        class_num_lst.append(class_mapping[class_name])

print(len(file_path_lst))
for i in range(10):
    print(f"File Path: {file_path_lst[i]}, Class: {class_num_lst[i]}")

       
        
print(len(file_path_lst))    
path_class_lst=list(zip(file_path_lst,class_num_lst))'''

# =================================================================================================================
# Making custom dataset
class custom_CIFAR10(Dataset):
  def __init__(self,root_dir, transform=None):
    self.root_dir=root_dir
    self.transform=transform
    
    self.class_name=sorted(os.listdir(root_dir))
    self.class_mapping={class_name: i for i, class_name in enumerate(self.class_name)}

    self.file_path_lst=[]
    self.class_num_lst=[]

    for class_name in self.class_name:
       class_dir = os.path.join(root_dir, class_name)
       for filename in os.listdir(class_dir):
          image_path=os.path.join(class_dir, filename)
          self.file_path_lst.append(image_path)
          self.class_num_lst.append(self.class_mapping[class_name])
    

  def __len__(self):
    return len(self.file_path_lst)


  def __getitem__(self, idx):
    image_path = self.file_path_lst[idx]
    image=Image.open(image_path).convert('RGB')
    class_num = self.class_num_lst[idx]

    if self.transform:
       image=self.transform(image)

    return image, class_num
  
  #img와 label을 return

data_transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(256),
    transforms.CenterCrop(224),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

train_file_path = 'C:/Users/pione/Desktop/IIPL/CIFAR/cifar10/cifar10/train'
train_CIFAR10 = custom_CIFAR10(root_dir=train_file_path, transform=data_transform)
train_loader = DataLoader(train_CIFAR10, batch_size=32, shuffle=True) #DataLoader

test_file_path = 'C:/Users/pione/Desktop/IIPL/CIFAR/cifar10/cifar10/test'
test_CIFAR10 = custom_CIFAR10(root_dir=test_file_path, transform=data_transform)
test_loader = DataLoader(test_CIFAR10, batch_size=32, shuffle=True) #DataLoader

def load_train():
  train_file_path = 'C:/Users/pione/Desktop/IIPL/CIFAR/cifar10/cifar10/train'
  train_CIFAR10 = custom_CIFAR10(root_dir=train_file_path, transform=data_transform)
  train_loader = DataLoader(train_CIFAR10, batch_size=32, shuffle=True) #DataLoader
  return train_loader

for images,labels in train_loader:
  break
#im=make_grid(images,nrow=16)

#plt.figure(figsize=(12,12))
#plt.imshow(np.transpose(im.numpy(),(1,2,0)))