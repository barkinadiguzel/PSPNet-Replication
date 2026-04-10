import torch.nn as nn
from encoder.encoder import Encoder
from blocks.pyramid_pooling import PyramidPoolingModule
from head.segmentation_head import SegmentationHead

class PSPNet(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()

        self.encoder = Encoder()

        self.ppm = PyramidPoolingModule(
            in_channels=2048,
            pool_sizes=(1, 2, 3, 6)
        )

        self.cls = SegmentationHead(
            in_channels=2048 + (2048 // 4) * 4,
            num_classes=num_classes
        )

    def forward(self, x):
        feat = self.encoder(x)             
        ppm_out = self.ppm(feat)          
        out = self.cls(ppm_out)             

        return out
