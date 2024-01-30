from functools import partial
import fastcore.all as fc
from matplotlib import pyplot as plt
import torch

from tinyai.cbs import Callback
from tinyai.core import to_cpu
from tinyai.viz import get_grid, show_image

__all__ = ["Hook", "Hooks", "HooksCallback", "ActivationStats"]


class Hook:
    def __init__(self, m, f):
        self.hook = m.register_forward_hook(partial(f, self))

    def remove(self):
        self.hook.remove()

    def __del__(self):
        self.remove()


class Hooks(list):
    def __init__(self, ms, f):
        super().__init__([Hook(m, f) for m in ms])

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()

    def __del__(self):
        self.remove()

    def __delitem__(self, i):
        self[i].remove()
        super().__delitem__(i)

    def remove(self):
        for h in self:
            h.remove()


class HooksCallback(Callback):
    def __init__(
        self, hookfunc, mod_filter=fc.noop, on_train=True, on_valid=False, mods=None
    ):
        self.hookfunc, self.mod_filter, self.on_train, self.on_valid, self.mods = (
            hookfunc,
            mod_filter,
            on_train,
            on_valid,
            mods,
        )
        super().__init__()

    def before_fit(self, learn):
        if self.mods:
            mods = self.mods
        else:
            mods = fc.filter_ex(learn.model.modules(), self.mod_filter)
        self.hooks = Hooks(mods, partial(self._hookfunc, learn))

    def _hookfunc(self, learn, *args, **kwargs):
        if (self.on_train and learn.training) or (self.on_valid and not learn.training):
            self.hookfunc(*args, **kwargs)

    def after_fit(self, learn):
        self.hooks.remove()

    def __iter__(self):
        return iter(self.hooks)

    def __len__(self):
        return len(self.hooks)


class ActivationStats(HooksCallback):
    def __init__(self, mod_filter=fc.noop):
        super().__init__(self.append_stats, mod_filter)

    @staticmethod
    def append_stats(hook, mod, inp, outp):
        if not hasattr(hook, "stats"):
            hook.stats = ([], [], [])
        acts = to_cpu(outp)
        hook.stats[0].append(acts.mean())
        hook.stats[1].append(acts.std())
        hook.stats[2].append(acts.abs().histc(40, 0, 10))

    @staticmethod
    def get_hist(h):
        return torch.stack(h.stats[2]).t().float().log1p()

    @staticmethod
    def get_min(h):
        h1 = torch.stack(h.stats[2]).t().float()
        return h1[0] / h1.sum(0)

    def color_dim(self, figsize=(11, 5)):
        fig, axes = get_grid(len(self), figsize=figsize)
        for ax, h in zip(axes.flat, self):
            show_image(self.get_hist(h), ax, origin="lower")

    def dead_chart(self, figsize=(11, 5)):
        fig, axes = get_grid(len(self), figsize=figsize)
        for ax, h in zip(axes.flatten(), self):
            ax.plot(self.get_min(h))
            ax.set_ylim(0, 1)

    def plot_stats(self, figsize=(10, 4)):
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        for h in self:
            for i in 0, 1:
                axs[i].plot(h.stats[i])
        axs[0].set_title("Means")
        axs[1].set_title("Stdevs")
        plt.legend(fc.L.range(self))

    def plot_all(self):
        self.color_dim()
        self.plot_stats()
        self.dead_chart()
