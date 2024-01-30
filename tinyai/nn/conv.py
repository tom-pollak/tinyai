from functools import partial
from typing import Any, Literal

from torch import nn
import torch.nn.functional as F


from tinyai.core import identity
from tinyai.nn.act import GeneralReLU
from tinyai.nn.init import init_params_

__all__ = [
    "conv",
    "deconv",
    "ResBlock",
    "ConvLayer",
    "Conv1x1",
    "Conv3x3",
    "LightConv3x3",
    "LightConvStream",
]

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


####


class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        IN=False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups,
        )
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        init_params_(self.modules())

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)


class Conv1x1(nn.Module):
    """1x1 convolution + bn/IN + relu."""

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        groups=1,
        IN=False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups,
        )

        if IN:
            self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.norm = nn.BatchNorm2d(out_channels)

        init_params_(self.modules())

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return F.relu(x)


class Conv3x3(nn.Module):
    """3x3 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        init_params_(self.modules())

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution.

    1x1 (linear) + dw 3x3 (nonlinear).
    """

    def __init__(self, in_channels, out_channels, dropout_p=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, padding=0, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            groups=out_channels,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if dropout_p is not None:
            self.drop = nn.Dropout2d(p=dropout_p)
        init_params_(self.modules())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = F.relu(x)
        if hasattr(self, "drop"):
            x = self.drop(x)
        return x


class LightConvStream(nn.Module):
    """Lightweight convolution stream."""

    def __init__(self, in_channels, out_channels, depth, dropout_p=None):
        super().__init__()
        assert depth >= 1, "depth must be equal to or larger than 1, but got {}".format(
            depth
        )
        layers = []
        layers += [LightConv3x3(in_channels, out_channels, dropout_p=dropout_p)]
        for i in range(depth - 1):
            layers += [LightConv3x3(out_channels, out_channels, dropout_p=dropout_p)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
