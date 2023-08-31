from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from custom_dataset import custom_CIFAR10

def load_data(train_dir, test_dir, data_transform):
  train_CIFAR10 = custom_CIFAR10(root_dir=train_dir, transform=data_transform)
  train_loader = DataLoader(train_CIFAR10, batch_size=32, shuffle=True) #DataLoader

  test_CIFAR10 = custom_CIFAR10(root_dir=test_dir, transform=data_transform)
  test_loader = DataLoader(test_CIFAR10, batch_size=32, shuffle=False) #DataLoader
  return train_loader,test_loader