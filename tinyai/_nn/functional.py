import torch
import torch.nn.functional as F

__all__ = ["cross_entropy"]

def cross_entropy(logits, target):
    "MPS crashes on cross entropy if passed single target tensor. I think this is because int64 is not supported"
    if target.ndim == 1:
        target = F.one_hot(target).to(torch.int32).float()
    return F.cross_entropy(logits, target)
