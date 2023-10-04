import torch.nn as nn
import torchvision.models as models

def model_res():
    model_res = models.resnet50()
    return model_res