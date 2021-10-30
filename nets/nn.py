import torch
import torch.nn as nn
from math import ceil

from utils import ConvBNAct, ConvBNSiLU, SEModule


class LinearBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, exp_ratio, stride, use_se=True, ratio=12):
        super(LinearBottleneck, self).__init__()
        self.shortcut = stride == 1 and in_channels <= out_channels
        self.in_channels = in_channels

        layers = []
        if exp_ratio != 1:
            dw_channels = in_channels * exp_ratio
            layers.append(ConvBNSiLU(in_channels=in_channels, out_channels=dw_channels))
        else:
            dw_channels = in_channels

        layers.append(ConvBNAct(in_channels=dw_channels, out_channels=dw_channels, kernel_size=3, stride=stride, padding=1, groups=dw_channels, act=False))

        if use_se:
            layers.append(SEModule(channels=dw_channels, ratio=ratio))

        layers.append(nn.ReLU6())
        layers.append(ConvBNAct(in_channels=dw_channels, out_channels=out_channels, act=False, relu6=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        if self.shortcut:
            out[:, 0:self.in_channels, :, :] += x

        return out


def _conf(in_channels=16, out_channels=180, width_mult=1.0, depth_mult=1.0, use_se=True, ratio=12):
    layers = [1, 2, 2, 3, 3, 5]
    strides = [1, 2, 2, 2, 1, 2]
    use_ses = [False, False, True, True, True, True]

    layers = [ceil(element * depth_mult) for element in layers]
    strides = sum([[element] + [1] * (layers[idx] - 1) for idx, element in enumerate(strides)], [])
    exp_ratios = [1] * layers[0] + [6] * sum(layers[1:])
    depth = sum(layers[:]) * 3

    if use_se:
        use_ses = sum([[element] * layers[idx] for idx, element in enumerate(use_ses)], [])
    else:
        use_ses = [False] * sum(layers[:])

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

    features.append(
        ConvBNSiLU(in_channels=3, out_channels=int(round(stem_channel * width_mult)), kernel_size=3, stride=2, padding=1))

    for idx, (in_ch, out_ch, exp_ratio, stride, se) in enumerate(
            zip(in_filers, out_filters, exp_ratios, strides, use_ses)):
        features.append(LinearBottleneck(in_channels=in_ch, out_channels=out_ch, exp_ratio=exp_ratio, stride=stride, use_se=se, ratio=ratio))

    pen_channels = int(1280 * width_mult)
    features.append(ConvBNSiLU(in_channels=out_filters[-1], out_channels=pen_channels))

    return features, pen_channels


class ReXNetV1(nn.Module):
    def __init__(self, width_mult=1.0, depth_mult=1.0, num_classes=1000, drop_rate=0.2):
        super(ReXNetV1, self).__init__()

        features, channels = _conf(width_mult=width_mult, depth_mult=depth_mult)
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Conv2d(channels, num_classes, kernel_size=1, bias=True),
            nn.Flatten())

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.mean((2, 3), keepdim=True))
        return x


def rexnet_100(width_mult=1.0, num_classes=1000):
    return ReXNetV1(width_mult=width_mult, num_classes=num_classes)


def rexnet_130(width_mult=1.3, num_classes=1000):
    return ReXNetV1(width_mult=width_mult, num_classes=num_classes)


def rexnet_150(width_mult=1.5, num_classes=1000):
    return ReXNetV1(width_mult=width_mult, num_classes=num_classes)


def rexnet_200(width_mult=2.0, num_classes=1000):
    return ReXNetV1(width_mult=width_mult, num_classes=num_classes)


def rexnet_220(width_mult=2.2, num_classes=1000):
    return ReXNetV1(width_mult=width_mult, num_classes=num_classes)


def rexnet_300(width_mult=3.0, num_classes=1000):
    return ReXNetV1(width_mult=width_mult, num_classes=num_classes)


if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)

    rexnet_100 = rexnet_100()
    rexnet_130 = rexnet_130()
    rexnet_150 = rexnet_150()
    rexnet_200 = rexnet_200()
    rexnet_220 = rexnet_220()
    rexnet_300 = rexnet_300()

    rexnet_100_features = rexnet_100.features
    rexnet_130_features = rexnet_130.features
    rexnet_150_features = rexnet_150.features
    rexnet_200_features = rexnet_200.features
    rexnet_220_features = rexnet_220.features
    rexnet_300_features = rexnet_300.features

    print('Num. of Params of RexNet V1 1.0: {}'.format(sum(p.numel() for p in rexnet_100.parameters() if p.requires_grad)))
    print('Num. of Params of RexNet V1 1.3: {}'.format(sum(p.numel() for p in rexnet_130.parameters() if p.requires_grad)))
    print('Num. of Params of RexNet V1 1.5: {}'.format(sum(p.numel() for p in rexnet_150.parameters() if p.requires_grad)))
    print('Num. of Params of RexNet V1 2.0: {}'.format(sum(p.numel() for p in rexnet_200.parameters() if p.requires_grad)))
    print('Num. of Params of RexNet V1 2.2: {}'.format(sum(p.numel() for p in rexnet_220.parameters() if p.requires_grad)))
    print('Num. of Params of RexNet V1 3.0: {}'.format(sum(p.numel() for p in rexnet_300.parameters() if p.requires_grad)))

    print('Output of RexNet V1 1.0: {}'.format(rexnet_100(x).shape))
    print('Output of RexNet V1 1.3: {}'.format(rexnet_130(x).shape))
    print('Output of RexNet V1 1.5: {}'.format(rexnet_150(x).shape))
    print('Output of RexNet V1 2.0: {}'.format(rexnet_200(x).shape))
    print('Output of RexNet V1 2.2: {}'.format(rexnet_220(x).shape))
    print('Output of RexNet V1 3.0: {}'.format(rexnet_300(x).shape))

    print('Feature Extractor Output of RexNet V1 1.0: {}'.format(rexnet_100_features(x).shape))
    print('Feature Extractor Output of RexNet V1 1.3: {}'.format(rexnet_130_features(x).shape))
    print('Feature Extractor Output of RexNet V1 1.5: {}'.format(rexnet_150_features(x).shape))
    print('Feature Extractor Output of RexNet V1 2.0: {}'.format(rexnet_200_features(x).shape))
    print('Feature Extractor Output of RexNet V1 2.2: {}'.format(rexnet_220_features(x).shape))
    print('Feature Extractor Output of RexNet V1 3.0: {}'.format(rexnet_300_features(x).shape))
