import torch
import torch.nn as nn
import torch.nn.functional as F

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6)):
        super().__init__()

        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=size),
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), 1, bias=False),
                nn.BatchNorm2d(in_channels // len(pool_sizes)),
                nn.ReLU(inplace=True)
            )
            for size in pool_sizes
        ])

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramids = [x]

        for stage in self.stages:
            out = stage(x)
            out = F.interpolate(out, size=(h, w),
                                mode="bilinear",
                                align_corners=True)
            pyramids.append(out)

        return torch.cat(pyramids, dim=1)
