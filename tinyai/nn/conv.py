from functools import partial
from typing import Any

from torch import nn


from tinyai.core import identity
from tinyai.nn.act import GeneralReLU

__all__ = ["conv", "deconv", "ResBlock"]

act_gr = partial(GeneralReLU, leak=0.1, sub=0.4)


def conv(fi, fo, ks=3, stride=2, act: Any = act_gr, norm=None, bias=None):
    if bias is None:
        bias = norm is None
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


def _conv_block(ni, nf, stride, act=act_gr, norm=None, ks=3):
    # if norm:
    #     # setting bn mean to 0, will 0 the second layer
    #     nn.init.zeros_(c2[1].weight)  # type: ignore
    return nn.Sequential(
        conv(ni, nf, stride=1, act=act, norm=norm, ks=ks),
        conv(nf, nf, stride=stride, act=None, norm=norm, ks=ks),
    )


class ResBlock(nn.Module):
    def __init__(self, fi, fo, stride=1, ks=3, act: Any = act_gr, norm=None):
        super().__init__()
        self.convs = _conv_block(fi, fo, stride, act, norm, ks)

        if fi != fo:
            self.id_conv = conv(fi, fo, ks=1, stride=1, act=None)
        else:
            self.id_conv = identity

        if stride != 1:
            self.pool = nn.AvgPool2d(2, ceil_mode=True)
        else:
            self.pool = identity

        self.act = act()

    def forward(self, x):
        return self.act(self.convs(x) + self.id_conv(self.pool(x)))
