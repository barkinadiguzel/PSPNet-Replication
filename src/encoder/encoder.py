import torch.nn as nn
from backbone.resnet_dilated import ResNetDilated

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNetDilated("resnet50")

    def forward(self, x):
        return self.backbone(x)
