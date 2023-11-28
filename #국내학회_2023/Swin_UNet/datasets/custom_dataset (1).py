from datasets.contrastive_viewgenerator import custom_transform
import os
import pandas as pd
from torchvision.io import read_image
import glob 
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import imageio, cv2
import pickle


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir=root_dir
        print(root_dir)
        self.custom_transform = custom_transform
        self.image_dir=os.path.join(root_dir,'img_slices')
        self.label_dir=os.path.join(root_dir,'label_slices')
        self.image_namelist = self.loadName(root_dir)
        # self.label_namelist = self.loadName(root_dir)[1]

    def loadName(self, root_dir):
        # self.root_dir=root_dir
        with open(os.path.join(self.root_dir,'image.txt'), 'rb') as f:
            namelist = pickle.load(f)
        # img_path_list=sorted(glob.glob(os.path.join(self.root_dir, "img_slices", "*.*")))
        # label_path_list = sorted(glob.glob(os.path.join(self.root_dir, "label_slices", "*.*")))
        # self.image_namelist=[]
        # self.label_namelist=[]
        # for path in img_path_list:
            # self.image_namelist.append(path.split('/')[-1])
        # for path in label_path_list:
            # self.label_namelist.append(path.split('/')[-1])
        # namelist=[self.image_namelist, self.label_namelist]


        return namelist

    def __len__(self):
        return len(self.image_namelist)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir,self.image_namelist[idx])
        image = cv2.imread(img_path)
        # image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        label_path = os.path.join(self.label_dir,self.image_namelist[idx])
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) 
        # numpy type

        if self.custom_transform:
            # print(image.shape)
            # print(label.shape)
            dict_list = self.custom_transform(image=image,mask=label)
            image_dic=dict_list[0] #{image,label}
            # print(image_dic['image'].shape)
            # print(image_dic['mask'].shape)
            pos_dic=dict_list[1] #{image, label}
            # print(pos_dic['image'].shape)
            # print(pos_dic['mask'].shape)



        # ToTensor
        return image_dic, pos_dic


        '''image-> 1 512 512 
        label -> 512 512
        pos 1 512 512'''



# train_ds=CustomDataset(root_dir='/home/work/CUAI6th_3/YoonjiLee/HCIK_2023/SwinNet/datasets/_3d_slices', transform=custom_transform)
# data_=train_ds.__getitem__(0)

# print(data_[0]['image'].shape)
# print(data_[0]['label'].shape)
# print(data_[1]['image'].shape)
# print(data_[1]['label'].shape)
# '''
# print(data_[0]['image'].shape)
# print(data_[0]['label'].shape)
# print(data_[1]['image'].shape)
# print(data_[1]['label'].shape)'''