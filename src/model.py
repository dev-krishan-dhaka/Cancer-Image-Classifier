
import torch, torch.nn as nn
from torchvision import models

def create_model(name: str, num_classes: int = 2, pretrained: bool = True):
    name = name.lower()
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        return m
    elif name == "resnet34":
        m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        return m
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unknown model: {name}")
