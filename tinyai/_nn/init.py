from __future__ import print_function
import torch
from torch import nn

from tinyai.core import to_cpu, cls_name, match_modules
from tinyai.hooks import Hook, Hooks

__all__ = ["init_params_", "LSUV_", "get_init_values"]


def init_params_(modules, leaky=0.0):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(
                m.weight, mode="fan_out", nonlinearity="leaky_relu", a=leaky
            )
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.InstanceNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def LSUV_(
    model,
    data,
    apply_only_to=("Conv", "Linear", "Bilinear"),
    std_tol=0.1,
    mean_tol=0.1,
    max_iters=10,
    do_ortho_init=True,
    logging_FN=print,
):
    r"""
    Applies layer sequential unit variance (LSUV), as described in
    `All you need is a good init` - Mishkin, D. et al (2015):
    https://arxiv.org/abs/1511.06422

    Args:
        model: `torch.nn.Module` object on which to apply LSUV.
        data: sample input data drawn from training dataset.
        apply_only_to: list of strings indicating target children
            modules. For example, ['Conv'] results in LSUV applied
            to children of type containing the substring 'Conv'.
        std_tol: positive number < 1.0, below which differences between
            actual and unit standard deviation are acceptable.
        max_iters: number of times to try scaling standard deviation
            of each children module's output activations.
        do_ortho_init: boolean indicating whether to apply orthogonal
            init to parameters of dim >= 2 (zero init if dim < 2).
        logging_FN: function for outputting progress information.

    Example:
        >>> model = nn.Sequential(nn.Linear(8, 2), nn.Softmax(dim=1))
        >>> data = torch.randn(100, 8)
        >>> LSUV_(model, data)
    """

    matched_modules = match_modules(model, apply_only_to)

    if do_ortho_init:
        logging_FN(
            f"Applying orthogonal init (zero init if dim < 2) to params in {len(matched_modules)} module(s)."
        )
        for m in matched_modules:
            for p in m.parameters():
                if p.dim() >= 2:
                    torch.nn.init.orthogonal_(p)
                else:
                    torch.nn.init.zeros_(p)

    logging_FN(
        f"Applying LSUV to {len(matched_modules)} module(s) (up to {max_iters} iters per module):"
    )

    def append_stat(hook, mod, inp, outp):
        d = outp.data
        hook.mean, hook.std = d.mean(), d.std()

    was_training = model.training
    model.train()  # sets all modules to training behavior
    with torch.no_grad():
        for i, m in enumerate(matched_modules):
            with Hook(m, append_stat) as hook:
                for t in range(max_iters):
                    _ = model(data)  # run data through model to get stats
                    m.weight.data /= hook.std + 1e-6  # type: ignore
                    if hasattr(m, "bias") and m.bias is not None:
                        m.bias.data -= hook.mean  # type: ignore
                    if abs(hook.mean) < mean_tol and abs(hook.std - 1.0) < std_tol:  # type: ignore
                        break
            logging_FN(
                f"Module {i:2} after {(t+1):2} itr(s) | {cls_name(m)} | Mean:{hook.mean:7.3f} | Std:{hook.std:6.3f}"  # type: ignore
            )

    if not was_training:
        model.eval()


def get_init_values(model, xb, layers=("Conv", "Linear", "Bilinear")):
    def append_stat(hook, mod, inp, outp):
        d = outp.data
        hook.mean, hook.std = d.mean().item(), d.std().item()

    modules = match_modules(model, layers)

    with Hooks(modules, append_stat) as hooks:
        model(xb)
        for i, hook in enumerate(hooks):
            print(
                f"Module {i:2} | {cls_name(hook.m)} | Mean:{hook.mean:7.3f} | Std:{hook.std:6.3f}"
            )
