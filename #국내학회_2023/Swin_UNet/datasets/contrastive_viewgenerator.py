'''
[code description] CustomDataset.py

contrastive learning view generator
code reference:
https://github.com/sthalles/SimCLR/blob/master/data_aug/view_generator.py
'''
import torch
from torch import nn
import numpy as np
from torch.nn import CrossEntropyLoss, CosineSimilarity
import glob
from monai.data import DataLoader, Dataset
from datasets._2d_transform import aug_transform

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, image, mask):
        # print(x.shape)
        # print(lable.shape)
        views = [self.base_transform(image=image, mask=mask) for i in range(self.n_views)]
        return views
# 서로 다른 두 augmentatino 이미지 [{img1:-, label1:-}, {img2:-, label2:-}]

custom_transform = ContrastiveLearningViewGenerator(base_transform=aug_transform)
