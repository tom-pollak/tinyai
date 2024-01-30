from __future__ import annotations
from functools import partial
import math
import fastcore.all as fc
from operator import attrgetter
import matplotlib.pyplot as plt
from fastprogress import progress_bar, master_bar

import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import default_collate
from torcheval.metrics import Mean


from tinyai.core import cls_name, def_device, identity, to_device, to_cpu
from tinyai.hooks import Hooks
from tinyai.nn.act import GeneralReLU
from tinyai.viz import show_image, get_grid


__all__ = [
    "CancelFitException",
    "CancelBatchException",
    "CancelEpochException",
    "run_cbs",
    "Callback",
    "ToDeviceCB",
    "TrainCB",
    "MetricsCB",
    "ProgressCB",
    "PlotCB",
    "PlotLossCB",
    "PlotMetricsCB",
    "EarlyStoppingCB",
    "SingleBatchCB",
    "OverfitSingleBatchCB",
    "LRFinderCB",
    "BatchTransformCB",
    "BaseSchedCB",
    "BatchSchedCB",
    "EpochSchedCB",
    "RecorderCB",
    "HooksCallback",
    "ActivationStats",
]


class CancelFitException(Exception):
    pass


class CancelBatchException(Exception):
    pass


class CancelEpochException(Exception):
    pass


def run_cbs(cbs, method_nm, learn=None, ignored=None):
    ignored = fc.L(ignored)
    for cb in sorted(cbs, key=attrgetter("order")):
        if ignored is not None and isinstance(cb, tuple(ignored)):
            continue
        method = getattr(cb, method_nm, None)
        if method is not None:
            method(learn)


def require_cbs(cbs, required):
    required = set(required)
    present_cbs = {type(cb) for cb in cbs if isinstance(cb, tuple(required))}
    missing_cbs = required - present_cbs
    if missing_cbs:
        raise ValueError(f"Requires callbacks: {', '.join(map(cls_name, missing_cbs))}")


class Callback:
    order = 0


## Training
class ToDeviceCB(Callback):
    order = 0

    def before_fit(self, learn):
        learn.model = learn.model.to(def_device)

    def before_batch(self, learn):
        learn.batch = to_device(learn.batch)


class TrainCB(Callback):
    order = 0

    def __init__(self, n_inp=1):
        self.n_inp = n_inp

    def predict(self, learn):
        learn.preds = learn.model(*learn.batch[: self.n_inp])

    def get_loss(self, learn):
        learn.loss = learn.loss_func(learn.preds, *learn.batch[self.n_inp :])

    def backward(self, learn):
        learn.loss.backward()

    def step(self, learn):
        learn.opt.step()

    def zero_grad(self, learn):
        learn.opt.zero_grad()


class MetricsCB(Callback):
    order = 1
    show_train = True

    def __init__(self, *ms, **metrics):
        for o in ms:
            metrics[cls_name(o)] = o
        self.metrics = metrics
        self.loss = Mean()

    def _log(self, log):
        print(log)

    def before_fit(self, learn):
        learn.metrics = self

    def before_epoch(self, learn):
        self.loss.reset()
        for o in self.metrics.values():
            o.reset()

    def after_batch(self, learn):
        x, y, *_ = to_cpu(learn.batch)
        self.loss.update(to_cpu(learn.loss), weight=len(x))
        for m in self.metrics.values():
            m.update(to_cpu(learn.preds), y)

    def after_epoch(self, learn):
        if (learn.training and self.show_train) or not learn.training:
            log = dict(
                epoch=learn.epoch,
                train="train" if learn.model.training else "eval",
                loss=f"{self.loss.compute().item():.4f}",
            )
            log.update(
                {k: f"{v.compute().item():.4f}" for k, v in self.metrics.items()}
            )
            self._log(log)


class ProgressCB(Callback):
    order = 2

    def before_fit(self, learn):
        learn.epochs = self.mbar = master_bar(learn.epochs)
        self.first = True
        if hasattr(learn, "metrics"):
            learn.metrics._log = self._log

    def _log(self, d):
        if self.first:
            self.mbar.write(list(d), table=True)
            self.first = False
        self.mbar.write(list(d.values()), table=True)

    def before_epoch(self, learn):
        learn.dl = progress_bar(learn.dl, leave=False, parent=self.mbar)

    def after_batch(self, learn):
        learn.dl.comment = f"{learn.loss:.3f}"


class PlotCB(Callback):
    order = 3

    def __init__(self):
        super().__init__()
        self.graph_kwargs = {}

    def after_batch(self, learn):
        if learn.training:  # store train
            self.mbar.update_graph(self._graph_data(), **self.graph_kwargs)  # type: ignore

    def after_epoch(self, learn):
        if not learn.training:  # store valid
            self.mbar.update_graph(self._graph_data(), **self.graph_kwargs)  # type: ignore

    def _graph_data(self):
        return []


class PlotLossCB(PlotCB):
    def __init__(self):
        super().__init__()
        self.required_cbs = [MetricsCB, ProgressCB]

    def before_fit(self, learn):
        print(learn.cbs)
        require_cbs(learn.cbs, self.required_cbs)
        self.mbar = learn.epochs
        self.train_losses = []
        self.val_losses = []

    def after_batch(self, learn):
        if learn.training:  # store train
            self.train_losses.append((learn.train_steps, learn.loss.item()))
        super().after_batch(learn)

    def after_epoch(self, learn):
        if not learn.training:  # store valid
            self.val_losses.append((learn.train_steps, learn.metrics.loss.compute()))
        super().after_epoch(learn)

    def _graph_data(self):
        trn_ls, val_ls = [], []
        if len(self.train_losses):
            trn_ls = [x.numpy() for x in default_collate(self.train_losses)]
        if len(self.val_losses):
            val_ls = [x.numpy() for x in default_collate(self.val_losses)]
        return [trn_ls, val_ls]


class PlotMetricsCB(PlotCB):
    def __init__(self):
        super().__init__()
        self.required_cbs = [MetricsCB]

    def before_fit(self, learn):
        require_cbs(learn.cbs, self.required_cbs)
        self.recs = {k: [] for k in learn.metrics.metrics.keys()}
        self.mbar = master_bar(range(learn.nepochs))
        self.mbar.names = list(learn.metrics.metrics.keys())
        self.graph_kwargs = {"x_bounds": (0,)}

    def after_batch(self, learn):
        pass  # override graph update

    def after_epoch(self, learn):
        if not learn.training:
            for k, m in learn.metrics.metrics.items():
                self.recs[k].append((learn.train_steps, m.compute()))
        return super().after_epoch(learn)

    def _graph_data(self):
        return [default_collate(x) for x in self.recs.values()]


# TODO: add option to save best model
# TODO: with early stopping enabled last epoch is not printed
class EarlyStoppingCB(MetricsCB):
    order = 3

    def __init__(self, patience=1, metric=None):
        "metric = None uses val loss"
        if metric is None:
            super().__init__()
            self.metric = self.loss
        else:
            super().__init__(es_metric=metric)
            self.metric = self.metrics["es_metric"]

        self.init_patience = patience
        self.patience = patience
        self.best_metric = float("inf")

    def after_epoch(self, learn):
        if not learn.training:
            metric = self.metric.compute().item()
            if metric > self.best_metric:
                self.patience -= 1
                if self.patience == 0:
                    print(
                        f"Early stopping {learn.epoch}: current {metric:.4f}, best {self.best_metric:.4f}"
                    )
                    raise CancelFitException()
            else:
                self.patience = self.init_patience
                self.best_metric = metric


class SingleBatchCB(Callback):
    order = 3

    def after_batch(self, learn):
        raise CancelEpochException()


class OverfitSingleBatchCB(SingleBatchCB):
    def __init__(self, eval_steps=float("inf")):
        self.eval_nsteps = eval_steps

    def after_batch(self, learn):
        # cancel after one training batch
        if learn.training:
            raise CancelEpochException()

    def before_batch(self, learn):
        # if validating, and not eval epoch, cancel validation
        if not learn.training and learn.train_steps % self.eval_nsteps != 0:
            raise CancelEpochException()


# TODO: model dosne't train correctly after lr_find
class LRFinderCB(Callback):
    order = 1

    def __init__(self, gamma=1.3, max_mult=3):
        super().__init__()
        self.gamma = gamma
        self.max_mult = max_mult

    def before_fit(self, learn):
        self.initial_state = learn.model.state_dict().copy()

        self.sched = ExponentialLR(learn.opt, gamma=self.gamma)
        self.lrs, self.losses = [], []
        self.min = float("inf")

    def before_epoch(self, learn):
        if not learn.training:
            raise CancelEpochException()

    def after_batch(self, learn):
        loss = to_cpu(learn.loss)
        self.lrs.append(learn.opt.param_groups[0]["lr"])
        self.losses.append(loss)
        if loss < self.min:
            self.min = loss
        if math.isnan(loss) or loss > self.min * self.max_mult:
            raise CancelFitException()
        self.sched.step()

    def cleanup_fit(self, learn):
        learn.opt.zero_grad()
        del self.sched
        learn.model.load_state_dict(self.initial_state)

        plt.plot(self.lrs, self.losses)
        plt.xscale("log")
        plt.show()


## Transforms
class BatchTransformCB(Callback):
    def __init__(self, tfm, train=True, val=True):
        self.tfm = tfm
        self.train = train
        self.val = val

    def before_batch(self, learn):
        if (self.train and learn.training) or (self.val and not learn.training):
            learn.batch = self.tfm(learn.batch)


## Scheduler
class BaseSchedCB(Callback):
    def __init__(self, sched_func):
        self.sched_func = sched_func

    def before_fit(self, learn):
        self.sched = self.sched_func(learn.opt)

    def _step(self, learn):
        if learn.training:
            self.sched.step()


class BatchSchedCB(BaseSchedCB):
    def after_batch(self, learn):
        self._step(learn)


class EpochSchedCB(BaseSchedCB):
    def after_epoch(self, learn):
        self._step(learn)


## Recorder
class RecorderCB(Callback):
    "holds results of named function"

    def __init__(self, **d):
        self.d = d

    def before_fit(self, learn):
        self.recs = {k: [] for k in self.d}

    def after_batch(self, learn):
        if not learn.training:
            return
        for k, v in self.d.items():
            self.recs[k].append(v(learn))

    def plot(self):
        for k, v in self.recs.items():
            plt.plot(v, label=k)
            plt.legend()
            plt.show()


##
class HooksCallback(Callback):
    def __init__(
        self, hookfunc, mod_filter=identity, on_train=True, on_valid=False, mods=None
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
            mods = filter(self.mod_filter, learn.model.modules())
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
    order = 4
    act_filter = lambda m: isinstance(
        m, (nn.ReLU, nn.LeakyReLU, nn.GELU, GeneralReLU, nn.Sigmoid)
    )

    def __init__(self, mod_filter=act_filter):
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
        fig.show()

    def dead_chart(self, figsize=(11, 5)):
        fig, axes = get_grid(len(self), figsize=figsize)
        for ax, h in zip(axes.flatten(), self):
            ax.plot(self.get_min(h))
            ax.set_ylim(0, 1)
        fig.show()

    def plot_stats(self, figsize=(10, 4)):
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        for h in self:
            for i in 0, 1:
                axs[i].plot(h.stats[i])
        axs[0].set_title("Means")
        axs[1].set_title("Stdevs")
        fig.legend(fc.L.range(self))
        fig.show()

    def plot_all(self):
        self.color_dim()
        self.plot_stats()
        self.dead_chart()
