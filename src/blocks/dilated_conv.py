import torch.nn as nn

class DilatedConv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=2):
        super().__init__()

        self.conv = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
