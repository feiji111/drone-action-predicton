from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B3_Weights,
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    efficientnet_b3,
    mobilenet_v3_large,
    mobilenet_v3_small,
    resnet18,
    resnet34,
    resnet50,
)


BACKBONES = {
    "mobilenet_v3_small": (mobilenet_v3_small, MobileNet_V3_Small_Weights),
    "mobilenet_v3_large": (mobilenet_v3_large, MobileNet_V3_Large_Weights),
    "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights),
    "efficientnet_b3": (efficientnet_b3, EfficientNet_B3_Weights),
    "resnet18": (resnet18, ResNet18_Weights),
    "resnet34": (resnet34, ResNet34_Weights),
    "resnet50": (resnet50, ResNet50_Weights),
}


def get_backbone(backbone_name: str, pretrained: bool = False) -> tuple[nn.Module, int]:
    backbone_name = backbone_name.lower()
    if backbone_name not in BACKBONES:
        raise ValueError(
            f"Unsupported backbone '{backbone_name}'. "
            f"Supported backbones: {sorted(BACKBONES)}"
        )

    build_fn, weights_enum = BACKBONES[backbone_name]
    weights = weights_enum.DEFAULT if pretrained else None

    try:
        backbone = build_fn(weights=weights)
    except TypeError:
        backbone = build_fn(pretrained=bool(pretrained))

    if backbone_name.startswith("mobilenet_v3"):
        feature_dim = backbone.classifier[0].in_features
        encoder = nn.Sequential(
            backbone.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )
    elif backbone_name.startswith("efficientnet"):
        feature_dim = backbone.classifier[1].in_features
        encoder = nn.Sequential(
            backbone.features,
            backbone.avgpool,
            nn.Flatten(1),
        )
    else:
        feature_dim = backbone.fc.in_features
        encoder = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten(1))

    return encoder, feature_dim
