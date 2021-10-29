import torch
import torch.nn as nn
from math import ceil
from utils import ConvBNAct, ConvBNSiLU, SE


class LinearBottleneck(nn.Module):
    def __init__(self, c1, c2, t, stride, use_se=True, ratio=12):
        super(LinearBottleneck, self).__init__()
        self.shortcut = stride == 1 and c1 <= c2
        self.in_channels = c1

        layers = []
        if t != 1:
            dw_channels = c1 * t
            layers.append(ConvBNSiLU(c1, dw_channels))
        else:
            dw_channels = c1

        layers.append(ConvBNAct(dw_channels, dw_channels, k=3, s=stride, p=1, g=dw_channels, act=False))

        if use_se:
            layers.append(SE(c1=dw_channels, c2=dw_channels, ratio=ratio))

        layers.append(nn.ReLU6())
        layers.append(ConvBNAct(dw_channels, c2, act=False, relu6=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        if self.shortcut:
            out[:, 0:self.in_channels, :, :] += x

        return out


class ReXNetV1(nn.Module):
    def __init__(self, in_channels=16, out_channels=180, width_mult=1.0, depth_mult=1.0, num_classes=1000,
                 use_se=True, ratio=12, dropout_ratio=0.2):
        super(ReXNetV1, self).__init__()

        layers = [1, 2, 2, 3, 3, 5]
        strides = [1, 2, 2, 2, 1, 2]
        use_ses = [False, False, True, True, True, True]

        layers = [ceil(element * depth_mult) for element in layers]
        strides = sum([[element] + [1] * (layers[idx] - 1)
                       for idx, element in enumerate(strides)], [])
        if use_se:
            use_ses = sum([[element] * layers[idx] for idx, element in enumerate(use_ses)], [])
        else:
            use_ses = [False] * sum(layers[:])
        ts = [1] * layers[0] + [6] * sum(layers[1:])

        depth = sum(layers[:]) * 3
        stem_channel = 32 / width_mult if width_mult < 1.0 else 32
        in_planes = in_channels / width_mult if width_mult < 1.0 else in_channels

        features = []
        in_filers = []
        out_filters = []

        # The following channel configuration is a simple instance to make each layer become an expand layer.
        for i in range(depth // 3):
            if i == 0:
                in_filers.append(int(round(stem_channel * width_mult)))
                out_filters.append(int(round(in_planes * width_mult)))
            else:
                in_filers.append(int(round(in_planes * width_mult)))
                in_planes += out_channels / (depth // 3 * 1.0)
                out_filters.append(int(round(in_planes * width_mult)))

        features.append(ConvBNSiLU(3, int(round(stem_channel * width_mult)), k=3, s=2, p=1))

        for block_idx, (in_c, c, t, s, se) in enumerate(zip(in_filers, out_filters, ts, strides, use_ses)):
            features.append(LinearBottleneck(c1=in_c, c2=c, t=t, stride=s, use_se=se, ratio=ratio))

        pen_channels = int(1280 * width_mult)
        features.append(ConvBNSiLU(c, pen_channels))

        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(dropout_ratio),
            nn.Conv2d(pen_channels, num_classes, 1, bias=True),
            nn.Flatten())

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def rexnetv1(width_mult=1.0, num_classes=1000):
    return ReXNetV1(width_mult=width_mult, num_classes=num_classes)


def rexnetv2(width_mult=2.0, num_classes=1000):
    return ReXNetV1(width_mult=width_mult, num_classes=num_classes)


if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)

    rexnetv1 = rexnetv1()
    rexnetv1_features = rexnetv1.features

    print('Num. of Params of RexNetV1: {}'.format(sum(p.numel() for p in rexnetv1.parameters() if p.requires_grad)))

    print('Output of RexNetV1: {}'.format(rexnetv1(x).shape))

    print('Feature Extractor Output of RexNetV1: {}'.format(rexnetv1_features(x).shape))
