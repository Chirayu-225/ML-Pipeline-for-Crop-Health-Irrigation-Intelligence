"""Multi-task leaf classifier: disease type (N classes) + severity (mild / moderate / severe)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """A compact custom CNN (the 'custom CNN' option; also used for CPU smoke tests)."""

    out_features = 128

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(_conv_block(3, 16), _conv_block(16, 32), _conv_block(32, 64), _conv_block(64, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).flatten(1)


class LeafClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_severity: int = 3,
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        if backbone == "resnet18":
            weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
            net = torchvision.models.resnet18(weights=weights)
            feat = net.fc.in_features
            net.fc = nn.Identity()
            self.backbone = net
        elif backbone == "small_cnn":
            self.backbone = SmallCNN()
            feat = SmallCNN.out_features
        else:
            raise ValueError(f"Unknown backbone {backbone!r}")
        self.feature_dim = feat
        self.drop = nn.Dropout(dropout)
        self.disease_head = nn.Linear(feat, num_classes)
        self.severity_head = nn.Linear(feat, num_severity)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.drop(self.backbone(x))
        return self.disease_head(f), self.severity_head(f)
