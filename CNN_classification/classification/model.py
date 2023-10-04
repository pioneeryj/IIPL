import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms, models

def model_build(model_name):
    if model_name=='resnet50':
        model = models.resnet50()
        data_transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(256),
                                           transforms.CenterCrop(224),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    elif model_name=='vit':
        model = models.vit_h_14()
        data_transform = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor(), 
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  ])
        
    return model, data_transform