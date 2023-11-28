import torch
from torchvision.transforms import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# data augmentation 
'''
albumentation augmentation techiques: 
https://github.com/albumentations-team/albumentations#i-want-to-explore-augmentations-and-see-albumentations-in-action
'''
aug_transform = A.Compose([
    # 이미지를 PyTorch tensor로 변환
    #A.augmentations.transforms.Equalize(mode='cv', by_channels=True, mask=None, mask_params=(), always_apply=False, p=0.5),
    #A.augmentations.transforms.RandomShadow(),
    #AAT.Sharpen(alpha=(0.2,0.5),lightness=(0.5, 1.0), always_apply=False, p=0.5),
    A.GaussianBlur(),
    #AAT.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.5),
    A.Rotate(),
    #A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),

    A.Resize(224,224),
    A.ToFloat(),
    #A.augmentations.transforms.PixelDropout(0.01),
    ToTensorV2(), 
])

# original_transform = transforms.ToTensor()

# transformed=aug_transform(image=image, mask=label)
# transformed_image = transformed['image']