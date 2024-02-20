from pathlib import Path
import random
from typing import Mapping, Any
import os

import matplotlib as mpl
import numpy as np

import torch
import torch.backends.mps
from torch.utils.data import default_collate

from IPython.core.getipython import get_ipython
from IPython.core.interactiveshell import InteractiveShell
from IPython.core.history import HistoryManager
import sys
import traceback
import gc

__all__ = [
    "MODEL_DIR",
    "list_models",
    "IMAGENET_STATS",
    "identity",
    "Noop",
    "noop",
    "cls_name",
    "set_output",
    "set_seed",
    "_num_cpus",
    "def_workers",
    "toggle_mpl_cmap",
    "match_modules",
    "get_children",
    "def_device",
    "to_device",
    "to_cpu",
    "collate_device",
    "clean_mem",
    "IN_NOTEBOOK",
]

MODEL_DIR = Path().home() / ".cache/tinyai/models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def list_models():
    return list(MODEL_DIR.glob("*.pth"))


IMAGENET_STATS = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

## UTILS


def identity(*x):
    return x


class Noop(torch.nn.Module):
    def forward(self, x):
        return x


def noop(x):
    Noop()(x)


def cls_name(cls):
    if isinstance(cls, type):
        return cls.__name__
    return type(cls).__name__


def set_output(precision=3):
    torch.set_printoptions(precision=precision, sci_mode=False, linewidth=140)
    mpl.rcParams["figure.constrained_layout.use"] = True


def set_seed(seed):
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _num_cpus() -> int:
    "Get number of cpus"
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()  # type: ignore


def_workers = _num_cpus()


def toggle_mpl_cmap():
    if mpl.rcParams["image.cmap"] == "viridis":
        mpl.rcParams["image.cmap"] = "gray_r"
    else:
        mpl.rcParams["image.cmap"] = "viridis"
    print("setting cmap:", mpl.rcParams["image.cmap"])


##


def get_children(model):
    children = list(model.children())
    return (
        [model]
        if len(children) == 0
        else [ci for c in children for ci in get_children(c)]
    )


def match_modules(model, layers: list[str]):
    return [
        m
        for m in get_children(model)
        if any(substr in cls_name(m) for substr in layers)
    ]


## Device

def_device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)


def to_device(x, device=def_device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, Mapping):
        return {k: v.to(device) for k, v in x.items()}
    return type(x)(o.to(device) for o in x)  # list, tuple, etc.


def to_cpu(x) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, Mapping):
        return {k: to_cpu(v) for k, v in x.items()}
    return type(x)(to_cpu(o) for o in x)


def collate_device(b):
    return to_device(default_collate(b))


## Clean Mem


def clean_ipython_hist():
    # Code in this function mainly copied from IPython source
    if not "get_ipython" in globals():
        return
    ip: InteractiveShell = get_ipython()  # type: ignore
    user_ns = ip.user_ns
    ip.displayhook.flush()
    pc = ip.displayhook.prompt_count + 1
    for n in range(1, pc):
        user_ns.pop("_i" + repr(n), None)
    user_ns.update(dict(_i="", _ii="", _iii=""))
    hm: HistoryManager = ip.history_manager  # type: ignore
    hm.input_hist_parsed[:] = [""] * pc
    hm.input_hist_raw[:] = [""] * pc
    hm._i = hm._ii = hm._iii = hm._i00 = ""


def clean_tb():
    if hasattr(sys, "last_traceback"):
        traceback.clear_frames(sys.last_traceback)
        delattr(sys, "last_traceback")
    if hasattr(sys, "last_type"):
        delattr(sys, "last_type")
    if hasattr(sys, "last_value"):
        delattr(sys, "last_value")


def clean_mem():
    clean_tb()
    clean_ipython_hist()
    gc.collect()
    torch.cuda.empty_cache()


# Cell
def in_notebook():
    "Check if the code is running in a jupyter notebook"
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":  # Jupyter notebook, Spyder or qtconsole
            import IPython

            return IPython.__version__ >= "6.0.0"
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


IN_NOTEBOOK = in_notebook()
