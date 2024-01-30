import random
from typing import Mapping
import fastcore.all as fc

import matplotlib as mpl
import numpy as np

import torch
import torch.backends
from torch.utils.data import default_collate

import sys
import traceback
import gc

def inplace(f):
    def _inner(x):
        f(x)
        return x

    return _inner

def num_params(model):
    return sum(o.numel() for o in model.parameters())

def set_output(precision=3):
    torch.set_printoptions(precision=precision, sci_mode=False, linewidth=140)
    mpl.rcParams["figure.constrained_layout.use"] = True

def set_seed(seed):
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

## Device

def_device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

def to_device(x, device=def_device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, Mapping):
        return {k: v.to(device) for k, v in x.items()}
    return type(x)(o.to(device) for o in x)  # list, tuple, etc.

def to_cpu(x):
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
    if not 'get_ipython' in globals(): return
    ip = get_ipython()
    user_ns = ip.user_ns
    ip.displayhook.flush()
    pc = ip.displayhook.prompt_count + 1
    for n in range(1, pc): user_ns.pop('_i'+repr(n),None)
    user_ns.update(dict(_i='',_ii='',_iii=''))
    hm = ip.history_manager
    hm.input_hist_parsed[:] = [''] * pc
    hm.input_hist_raw[:] = [''] * pc
    hm._i = hm._ii = hm._iii = hm._i00 =  ''

def clean_tb():
    if hasattr(sys, 'last_traceback'):
        traceback.clear_frames(sys.last_traceback)
        delattr(sys, 'last_traceback')
    if hasattr(sys, 'last_type'): delattr(sys, 'last_type')
    if hasattr(sys, 'last_value'): delattr(sys, 'last_value')

def clean_mem():
    clean_tb()
    clean_ipython_hist()
    gc.collect()
    torch.cuda.empty_cache()

