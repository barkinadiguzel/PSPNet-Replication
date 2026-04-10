import torchvision.models as models
import torch.nn as nn

class ResNetDilated(nn.Module):
    def __init__(self, backbone="resnet50"):
        super().__init__()

        resnet = getattr(models, backbone)(pretrained=True)

        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self._make_dilated(self.layer3, dilation=2)
        self._make_dilated(self.layer4, dilation=4)

    def _make_dilated(self, layer, dilation):
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                if m.stride == (2, 2):
                    m.stride = (1, 1)
                m.dilation = (dilation, dilation)
                m.padding = (dilation, dilation)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
