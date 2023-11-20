from contrastive_viewgenerator import ContrastiveLearningViewGenerator
from monai.data import DataLoader, Dataset
from data_transform import train_files, generat_transforms


# The size of the images
output_shape = [224,224]
kernel_size = [21,21] # 10% of the output_shape

# The custom transform
custom_transform = ContrastiveLearningViewGenerator(base_transform=generat_transforms)
# image[idx]=[img1, img2]
train_ds=Dataset(data=train_files, transform=custom_transform)
train_dl = DataLoader(train_ds, batch_size=1)