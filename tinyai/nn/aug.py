import numpy as np
import torch
from torch import nn

# from torch.nn import init


__all__ = ["random_erase", "RandomErase"]


def _erase_block(x, pct, bmean, bstd, bmin, bmax):
    h, w = x.shape[-2], x.shape[-1]
    szx, szy = int(pct * h), int(pct * w)
    stx, sty = np.random.randint(0, h - szx), np.random.randint(0, w - szy)

    noise = torch.normal(mean=bmean, std=bstd, size=(szx, szy))
    x[:, :, stx : stx + szx, sty : sty + szy] = noise

    x = x.clamp(bmin, bmax)
    return x


def random_erase(x, pct=0.2, max_n=4):
    bmean, bstd, bmin, bmax = x.mean(), x.std(), x.min(), x.max()
    for _ in range(np.random.randint(0, max_n)):
        _erase_block(x, pct, bmean, bstd, bmin, bmax)
    return x


class RandomErase(nn.Module):
    def __init__(self, pct=0.2, max_n=4):
        super().__init__()
        self.pct = pct
        self.max_n = max_n

    def forward(self, x):
        return random_erase(x, self.pct, self.max_n)
