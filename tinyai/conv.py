from torch import nn


def conv(fi, fo, ks=3, stride=2, act=True):
    c = nn.Conv2d(fi, fo, kernel_size=ks, stride=stride, padding=ks // 2)
    if act:
        return nn.Sequential(c, nn.ReLU())
    return c


def deconv(fi, fo, ks=3, act=True):
    layers = [
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.Conv2d(fi, fo, kernel_size=ks, stride=1, padding=ks // 2),
    ]
    if act:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)
