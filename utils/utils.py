import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True, relu6=True):
        super(ConvBNAct, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=k, stride=s, padding=p, groups=g, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU6() if act and relu6 else nn.ReLU())

    def forward(self, x):
        return self.conv(x)


class ConvBNSiLU(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1):
        super(ConvBNSiLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=k, stride=s, padding=p, groups=g, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU())

    def forward(self, x):
        return self.conv(x)


class SE(nn.Module):
    def __init__(self, c1, c2, ratio=12):
        super(SE, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2 // ratio, kernel_size=1, padding=0),
            nn.BatchNorm2d(c2 // ratio),
            nn.ReLU(),
            nn.Conv2d(c2 // ratio, c2, kernel_size=1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        return x * self.conv(self.pool(x))
