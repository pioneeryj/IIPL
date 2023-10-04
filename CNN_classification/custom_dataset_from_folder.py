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

# making custom dataset
# =================================================================================================================

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
