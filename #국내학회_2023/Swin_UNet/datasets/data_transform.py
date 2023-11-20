
'''
[code description] data_transform.py

get synapse raw data and apply transformation
original_ds<- train_dataset applied with original transformation
generat_ds<-train_dataset applied with generat transformation(augmentation)

code reference:
https://pycad.co/3d-volumes-augmentation-for-tumor-segmentation/

'''
import monai
# monai: it is an open source framework based on Pytorch that can be used to segment or classify medical images.
from monai.utils import first
from monai.transforms import (
    Compose,
    CropForegroundd,
    LoadImaged,
    LoadImage,
    Orientationd,
    RandSpatialCropd,
    ScaleIntensityRanged,
    Spacingd,
    Resized,
    ToTensord,
    ScaleIntensityd,
    RandRotated,
    RandZoomd,
    RandGaussianNoised,
    EnsureChannelFirstd,
    Flipd,
    RandAffined,
)
from monai.data import DataLoader, Dataset
import monai.visualize
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import os
import torch
import nibabel as nib
from tqdm import tqdm

# check data if num of images and labels are the same
img_path='C:/Users/pione/Desktop/IIPL/#국내학회_2023/data/Synapse_raw/RawData/Training/img'
files=os.listdir(img_path)
num_files_img = len(files)
print(f"image data shape: {num_files_img}")

file_1='C:/Users/pione/Desktop/IIPL/#국내학회_2023/data/Synapse_raw/RawData/Training/img/img0001.nii.gz'
# train_file
data_dir='C:/Users/pione/Desktop/IIPL/#국내학회_2023/data/Synapse_raw/RawData/Training'.replace("/","\\")
train_images = sorted(glob.glob(os.path.join(data_dir, "img", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "label", "*.nii.gz")))
train_data = LoadImage(image_only=True, ensure_channel_first=True, simple_keys=True)(train_images[1])

# check data shape: torch.Size([1, 512, 512, 147])
# (batch size, height, width, slices)
print(f"image data shape: {train_data.shape}")


fig = monai.visualize.matshow3d(monai.transforms.Orientation("SPL")(train_data), every_n=5)

print(train_images[1]) #path 확인( \ 또는 / 이상하지 않은지)

# train_files<-make it dictionary: [image,label]
train_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
print(f"data length: {len(train_files)}")

loader=LoadImaged(keys=("image", "label"), image_only=False)
train_dic=loader(train_files[0])

# 원본 데이터 샘플 시각화
image, label = train_dic["image"], train_dic["label"]
plt.figure("visualize", (8,4))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(image[:, :, 60], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("label")
plt.imshow(label[:, :, 60])
plt.show()


# channel-first 해주기
ensure_channel_first = EnsureChannelFirstd(keys=["image", "label"])
datac_dict = ensure_channel_first(train_files)
print(f"image shape: {datac_dict['image'].shape}")
print(f"label shape: {datac_dict['label'].shape}")


original_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200,b_min=0.0, b_max=1.0, clip=True,), 
        ToTensord(keys=["image", "label"]),
    ]
)

generat_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="XY"),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200,b_min=0.0, b_max=1.0, clip=True,), 
        RandAffined(keys=['image', 'label'], prob=0.5, translate_range=10), 
        RandRotated(keys=['image', 'label'], prob=0.5, range_x=10.0),
        RandGaussianNoised(keys='image', prob=0.5),
        ToTensord(keys=["image", "label"]),
    ]
)

original_ds=Dataset(data=train_files, transform=original_transforms)
original_loader = DataLoader(original_ds, batch_size=1)
original_patient = first(original_loader)

generat_ds=Dataset(data=train_files, transform=generat_transforms)
generat_loader = DataLoader(generat_ds, batch_size=1)
generat_patient = first(generat_loader)

# show slice#30 of first patient
number_slice = 50
plt.figure("display", (12, 6))
plt.subplot(1, 2, 1)
plt.title(f"Original patient slice {number_slice}")
plt.imshow(original_patient["image"][0, 0, :, :, number_slice], cmap="gray")
# [batch,channel,H,W,num_slice]
plt.subplot(1, 2, 2)
plt.title(f"Generated patient slice {number_slice}")
plt.imshow(generat_patient["image"][0, 0, :, :, number_slice], cmap="gray")


# 앞뒤로 15개 빼고 slice sampling