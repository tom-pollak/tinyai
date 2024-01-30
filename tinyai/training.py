import torch
import torch.nn.functional as F
from torch import nn

from tinyai.core import to_cpu
from tinyai.hooks import Hook

__all__ = [
    "cross_entropy",
    "init_weights",
    "GeneralReLU",
    "lsuv_init",
]


def cross_entropy(logits, target):
    "MPS crashes on cross entropy if passed single target tensor. I think this is because int64 is not supported"
    if target.ndim == 1:
        target = F.one_hot(target).to(torch.int32).float()
    return F.cross_entropy(logits, target)


def init_weights(m, leaky=0.0):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, a=leaky)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class GeneralReLU(nn.Module):
    def __init__(self, leak=None, sub=None, maxv=None):
        super().__init__()
        self.leak, self.sub, self.maxv = leak, sub, maxv

    def forward(self, x):
        x = F.leaky_relu(x, self.leak) if self.leak is not None else F.relu(x)
        if self.sub is not None:
            x -= self.sub
        if self.maxv is not None:
            x.clamp_max_(self.maxv)
        return x


def _lsuv_stats(hook, mod, inp, outp):
    acts = to_cpu(outp)
    hook.mean = acts.mean()
    hook.std = acts.std()


def lsuv_init(model, m, m_in, xb, tol=1e-3, max_attempts=50):
    "https://arxiv.org/abs/1511.06422"
    h = Hook(m, _lsuv_stats)
    for i in range(max_attempts):
        with torch.no_grad():
            model(xb)
        if abs(h.mean) < tol and abs(1 - h.std) < tol:  # type:ignore
            break
        m_in.bias -= h.mean  # type: ignore
        m_in.weight.data /= h.std  # type: ignore
    h.remove()
    return i  # type: ignore
