# if문으로 custom_data 로드하기
# data가 CIFAR10 일때와 mnist일때

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms





class Custom_Dataset(Dataset):
    def __init__(self, root_dir,train_bool):
      if root_dir=="cifar_dir":
         dataset=datasets.CIFAR10(root=root_dir, train=train_bool,download=True, transform=None)

      elif root_dir=="mnist_dir":
         dataset=datasets.MNIST(root=root_dir, train=train_bool,download=True, transform=None)
      
      self.root_dir=root_dir
      self.dataset=dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):
        image, label = self.dataset[idx]
        return image, label
    
    #torchvision에서 데이터를 불러오면 데이터의 구조를 알수 있는 방법은 구글링뿐?