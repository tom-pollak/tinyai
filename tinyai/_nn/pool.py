from torch import nn

__all__ = ["GlobalAvgPool"]

class GlobalAvgPool(nn.Module):
    def forward(self, x):
        # inp: BCHW, mean over HW -> BC
        return x.mean((-1, -2))
