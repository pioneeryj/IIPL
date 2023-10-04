from torch.utils.data import Dataset
from torch.utils.data import DataLoader,random_split
from torchvision import datasets, transforms, models
from custom_dataset_from_folder import custom_CIFAR10
import custom_dataset_Dataset
from custom_dataset_Dataset import Custom_Dataset

cifar_dir='./cifar_data'
mnist_dir='./mnist_data'

def load_data(root_dir, data_transform):
  if root_dir=='cifar_dir':
    train_CIFAR10 = Custom_Dataset(root_dir=root_dir, train_bool=True, transform=data_transform)

    train_len=len(train_CIFAR10)
    train_size=int(0.8*train_len)
    val_size=train_len-train_size
    train_CIFAR10, valid_CIFAR10=random_split(train_CIFAR10,[train_size,val_size])
    test_CIFAR10 = Custom_Dataset(root_dir=root_dir, train_bool=False, transform=data_transform)

    train_loader = DataLoader(train_CIFAR10, batch_size=32, shuffle=True) #DataLoader
    test_loader = DataLoader(test_CIFAR10, batch_size=32, shuffle=False)
    valid_loader = DataLoader(valid_CIFAR10,batch_size=32, shuffle=False)
    dataset=datasets.cifar10(root=root_dir, train=True, transform=None)
    class_names=dataset.classes

  elif root_dir=='mnist_dir':
    train_MNIST = Custom_Dataset(root_dir=root_dir, train_bool=True, transform=data_transform)

    train_len=len(train_MNIST)
    train_size=int(0.8*train_len)
    val_size=train_len-train_size
    train_MNIST, valid_MNIST=random_split(train_MNIST,[train_size,val_size])

    test_MNIST = Custom_Dataset(root_dir=root_dir, train_bool=False, transform=data_transform)
    train_loader = DataLoader(train_MNIST, batch_size=32, shuffle=True) #DataLoader
    test_loader = DataLoader(test_MNIST, batch_size=32, shuffle=False)
    valid_loader = DataLoader(valid_MNIST,batch_size=32, shuffle=False)
    dataset=datasets.mnist(root=root_dir, train=True, transform=None)
    class_names=dataset.classes

  
  return train_loader, valid_loader, test_loader, class_names

# mnist 이미 transfrom이 적용되어 있는건가?
# class_num 변수를 만드는 방식이 비효율적임(custom_data 코드 반복)

