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
import custom_dataset,data_loader,model,engine
from model import model_build
from data_loader import load_data
from custom_dataset import custom_CIFAR10
import utils
from utils import save_model

# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.05

# Setup directories
train_dir = "C:/Users/pione/Desktop/IIPL/CIFAR/cifar10/cifar10/train"
test_dir = "C:/Users/pione/Desktop/IIPL/CIFAR/cifar10/cifar10/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(256),
    transforms.CenterCrop(224),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

train_dataloader, test_dataloader=data_loader.load_data(train_dir=train_dir, test_dir=test_dir, data_transform=data_transform, batch_size=BATCH_SIZE)

# model
model_res = model.model_build().to(device)

#set loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0001)

engine.train(model=model_res, train_dataloader=train_dataloader,
             test_dataloader=test_dataloader, loss_fn=loss_fn,
             optimizer=optimizer, epochs=NUM_EPOCHS, device=device)

# save the model
target_dir="C:/Users/pione/Desktop/IIPL/CIFAR_classification"
model_name="resnet50_for_CIFAR_classification.pth"
utils.save_model(model=model_res, target_dir=target_dir,
                 model_name=model_name)

# model load
model_res.load_state_dict(torch.load(target_dir/model_name))
