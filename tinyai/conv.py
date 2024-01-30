from typing import Any
from torch import nn

__all__ = ["conv", "deconv"]


def conv(fi, fo, ks=3, stride=2, act: Any = nn.ReLU, bias=True, norm=None):
    layers = [
        nn.Conv2d(fi, fo, kernel_size=ks, stride=stride, padding=ks // 2, bias=bias)
    ]
    if norm is not None:
        layers.append(norm(fo))
    if act is not None:
        layers.append(act())
    return nn.Sequential(*layers)


def deconv(fi, fo, ks=3, act=True):
    layers = [
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.Conv2d(fi, fo, kernel_size=ks, stride=1, padding=ks // 2),
    ]
    if act:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)
