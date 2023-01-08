import torch
import torch.nn as nn
from torchvision import models, transforms

"""
This file defines the model we will be using
We are creating varient of resnet, by loading the public model 
and changing the last few layers.
"""

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
}

def initialize_model(num_classes):
    model = models.resnet18(weights='DEFAULT')
    model.requires_grad_(False)

    num_ftrs = model.fc.in_features
    # replace the last FC layer
    model.fc = nn.Linear(num_ftrs, num_classes) # replace the last FC layer
    
    pretrained_layers = (list(model.children()))[:-2]

    combined_model = nn.Sequential(
        *pretrained_layers,
        model.layer4,
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Linear(num_ftrs, num_classes),
        nn.Sigmoid()
        )

    return combined_model