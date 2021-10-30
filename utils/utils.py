import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, act=True, relu6=True):
        super(ConvBNAct, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU6(inplace=True) if act and relu6 else nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class ConvBNSiLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1):
        super(ConvBNSiLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.SiLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class SEModule(nn.Module):
    def __init__(self, channels, ratio=12):
        super(SEModule, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(channels, channels // ratio, 1),
                                  nn.ReLU(inplace=True),
                                  nn.BatchNorm2d(channels // ratio),
                                  nn.Conv2d(channels // ratio, channels, 1),
                                  nn.Sigmoid())

    def forward(self, x):
        return x * self.conv(x.mean((2, 3), keepdim=True))
