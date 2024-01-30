import torch
from torch import nn

from tinyai.core import to_cpu
from tinyai.hooks import Hook

__all__ = [
    "init_weights",
    "lsuv_init",
]


# def init_weights(m, leaky=0.0):
#     if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
#         nn.init.kaiming_normal_(m.weight, a=leaky)
#         # if m.bias is not None:
#         #     nn.init.zeros_(m.bias)


def init_weights(m, leaky=0.0):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(m.weight, a=leaky)


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
