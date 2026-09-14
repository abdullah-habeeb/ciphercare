import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes, pretrained=True):
    # Use standard ResNet50
    # User requested: model = torchvision.models.resnet50(pretrained=True)
    # Note: 'pretrained' is deprecated in newer versions for 'weights', but we stick to user request or compatible
    try:
        model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
    except:
        model = models.resnet50(pretrained=pretrained)
    
    # Modify FC layer
    # ResNet50 fc in features is 2048
    model.fc = nn.Linear(2048, num_classes)
    
    return model
