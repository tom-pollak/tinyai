from typing import Any

from torch import nn
from torch.nn import init


from tinyai.core import identity

__all__ = ["conv", "deconv", "ResBlock"]


def conv(fi, fo, ks=3, stride=2, act: Any = nn.ReLU, norm=None, bias=True):
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


def _conv_block(fi, fo, stride, ks=3, act=nn.ReLU, norm=nn.BatchNorm2d):
    c1 = conv(fi, fo, ks, stride=1, act=act, norm=norm)
    c2 = conv(fo, fo, ks, stride=stride, act=None, norm=norm)
    if norm:
        # setting bn mean to 0, will 0 the second layer
        nn.init.zeros_(c2[1].weight)  # type: ignore
    return nn.Sequential(c1, c2)


class ResBlock(nn.Module):
    def __init__(self, fi, fo, stride=1, ks=3, act: Any = nn.ReLU, norm=nn.BatchNorm2d):
        super().__init__()
        self.convs = _conv_block(fi, fo, stride, ks, act, norm)
        if stride == 1:
            self.id_conv = identity
            self.pool = identity
        elif stride == 2:
            self.id_conv = conv(fi, fo, stride=1, ks=1, act=None, norm=None)
            self.pool = nn.AvgPool2d(2, ceil_mode=True)
        else:
            raise ValueError("stride must be either 1 or 2, given:", stride)
        self.act = act()

    def forward(self, x):
        return self.act(self.convs(x) + self.id_conv(self.pool(x)))
