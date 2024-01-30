from __future__ import annotations
from copy import deepcopy
import os
from functools import partial
import math
import fastcore.all as fc
from operator import attrgetter
import matplotlib.pyplot as plt
from fastprogress import progress_bar, master_bar
from datetime import datetime
import logging

import torch
from torch import Tensor
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import default_collate
from torcheval.metrics import Mean
from accelerate import Accelerator


from tinyai.core import cls_name, def_device, get_children, to_device, to_cpu, Noop
from tinyai.hooks import Hooks
from tinyai.nn.act import GeneralReLU
from tinyai.viz import show_image, get_grid


__all__ = [
    "CancelFitException",
    "CancelBatchException",
    "CancelEpochException",
    "run_cbs",
    "Callback",
    "DeviceCB",
    "TrainCB",
    "MetricsCB",
    "ProgressCB",
    "PlotCB",
    "PlotLossCB",
    "PlotMetricsCB",
    "EarlyStoppingCB",
    "CheckpointCB",
    "NBatchCB",
    "OverfitBatch",
    "LRFinderCB",
    "BatchTransformCB",
    "BaseSchedCB",
    "BatchSchedCB",
    "EpochSchedCB",
    "RecorderCB",
    "HooksCallback",
    "ActivationStats",
    "AccelerateCB",
]


class CancelFitException(Exception):
    pass


class CancelBatchException(Exception):
    pass


class CancelEpochException(Exception):
    pass


def run_cbs(cbs, method_nm, learn, ignored=None):
    for cb in sorted(cbs, key=attrgetter("order")):
        if ignored is not None and isinstance(cb, tuple(ignored)):
            continue
        method = getattr(cb, method_nm, None)
        if method is not None:
            method(learn)


def require_cbs(cbs, required_cbs):
    required_cbs = deepcopy(required_cbs)
    for cb in cbs:
        for reqcb in required_cbs:
            if isinstance(cb, reqcb):
                required_cbs.remove(reqcb)

    if len(required_cbs):
        raise ValueError(f"Required callback {required_cbs}")


def no_callbacks(cbs, no_cbs):
    for cb in cbs:
        for nocb in no_cbs:
            if isinstance(cb, nocb):
                raise ValueError(f"Callback {nocb} not allowed")


class Callback:
    order = 0


## Training
class DeviceCB(Callback):
    order = 0

    def before_fit(self, learn):
        learn.model = learn.model.to(def_device)

    def before_batch(self, learn):
        learn.batch = to_device(learn.batch)


class BaseTrainCB(Callback):
    order = 10

    ## Running mean of epoch loss
    def before_fit(self, learn):
        learn.epoch_loss = Mean()

    def before_epoch(self, learn):
        learn.epoch_loss.reset()

    def after_batch(self, learn):
        x, y, *_ = to_cpu(learn.batch)
        learn.epoch_loss.update(to_cpu(learn.loss), weight=len(x))


class TrainCB(BaseTrainCB):
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
    order = 50
    sigmoid = False
    show_train = True

    required_cbs = [BaseTrainCB]

    def __init__(self, *ms, plot=False, **metrics):
        for o in ms:
            metrics[cls_name(o)] = o
        self.metrics = metrics

    def _log(self, log):
        print(log)

    def before_fit(self, learn):
        learn.metrics = self

    def before_epoch(self, learn):
        for o in self.metrics.values():
            o.reset()

    def after_batch(self, learn):
        _, y, *_ = to_cpu(learn.batch)

        preds = to_cpu(learn.preds)
        if self.sigmoid:
            preds = preds.sigmoid()

        for m in self.metrics.values():
            m.update(preds, y)

    def after_epoch(self, learn):
        if (learn.training and self.show_train) or not learn.training:
            log = dict(
                epoch=learn.epoch,
                train="train" if learn.model.training else "eval",
                loss=f"{learn.epoch_loss.compute().item():.4f}",
            )
            for k, v in self.metrics.items():
                metric = v.compute()
                if isinstance(metric, Tensor):
                    if metric.numel() == 1:
                        log.update({k: f"{metric.item():.4f}"})
                    elif metric.ndim == 1:
                        log.update(
                            {
                                f"{k}_{i}": f"{o.item():.4f}"
                                for i, o in enumerate(metric)
                            }
                        )
                    else:
                        raise ValueError(f"{metric.shape} is not compatiable")
                elif isinstance(metric, tuple):  # multi output, (tensor, tensor)
                    log.update(
                        {
                            f"{k}": ";".join(f"{o.item():.4f}" for o in ms)
                            for ms in zip(*metric)
                        }
                    )
                else:
                    raise ValueError(f"{metric} is not compatiable")
            self._log(log)


class ProgressCB(Callback):
    order = MetricsCB.order + 1

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
    order = ProgressCB.order + 1

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

    def cleanup_fit(self, learn):
        plt.close()


class PlotLossCB(PlotCB):
    def __init__(self):
        super().__init__()
        self.required_cbs = [BaseTrainCB, ProgressCB]

    def before_fit(self, learn):
        require_cbs(learn.cbs, self.required_cbs)
        self.mbar = learn.epochs
        self.train_losses = []
        self.val_losses = []

    def after_batch(self, learn):
        if learn.training:  # store train
            self.train_losses.append((learn.train_steps, to_cpu(learn.loss)))
        super().after_batch(learn)

    def after_epoch(self, learn):
        if not learn.training:  # store valid
            self.val_losses.append(
                (learn.train_steps, to_cpu(learn.epoch_loss.compute()))
            )
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


class EarlyStoppingCB(MetricsCB):
    order = 100

    def __init__(self, patience=1, metric=None):
        "metric = None uses val loss"
        if metric is not None:
            super().__init__(es_metric=metric)
            self.metric = self.metrics["es_metric"]

        self.init_patience = patience
        self.patience = patience
        self.best_metric = float("inf")

    def before_fit(self, learn):
        if self.metric is None:
            self.metric = learn.epoch_loss

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


class CheckpointCB(MetricsCB):
    order = EarlyStoppingCB.order - 1

    def __init__(self, metric=None):
        "metric = None uses val loss"
        if metric is not None:
            super().__init__(ckpt_metric=metric)
            self.metric = self.metrics["ckpt_metric"]

        self.best_metric = float("inf")

    def before_fit(self, learn):
        if self.metric is None:
            self.metric = learn.epoch_loss

    def after_epoch(self, learn):
        if not learn.training:
            metric = self.metric.compute().item()
            if metric > self.best_metric:
                logging.info(f"Checkpoint {learn.epoch}: {metric:.4f}")
                fn = f"ckpt-{learn.epoch}_{datetime.now().strftime('%Y-%m-%d-%H:%M')}_{cls_name(learn.model)}"
                self.best_metric = metric
                learn.save(fn, overwrite=True)


class NBatchCB(Callback):
    order = 100

    def __init__(self, nbatches=1):
        self.nbatches = nbatches

    def after_batch(self, learn):
        if learn.iter + 1 >= self.nbatches:
            raise CancelEpochException()


class OverfitBatch(Callback):
    order = 100

    def __init__(self, nbatches=1, eval_steps=1):
        self.eval_nsteps = eval_steps
        self.nbatches = nbatches

    def after_batch(self, learn):
        # cancel after nbatches training batches if training
        if learn.training and learn.iter + 1 >= self.nbatches:
            raise CancelEpochException()

    def before_epoch(self, learn):
        # if validating, and not eval epoch, cancel validation
        if not learn.training and learn.train_steps % self.eval_nsteps != 0:
            raise CancelEpochException()


class LRFinderCB(Callback):
    order = TrainCB.order + 1

    def __init__(self, gamma=1.3, max_mult=3):
        super().__init__()
        self.gamma = gamma
        self.max_mult = max_mult

    def before_fit(self, learn):
        torch.save(learn.model, ".lr_tmp.pkl")

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
        learn.model = torch.load(".lr_tmp.pkl")
        os.remove(".lr_tmp.pkl")

        plt.plot(self.lrs, self.losses)
        plt.xscale("log")
        plt.show()


## Transforms
class BatchTransformCB(Callback):
    order = DeviceCB.order + 1

    def __init__(self, tfm, train=True, valid=True):
        self.tfm = tfm
        self.train = train
        self.valid = valid

    def before_batch(self, learn):
        if (self.train and learn.training) or (self.valid and not learn.training):
            learn.batch = self.tfm(learn.batch)


## Scheduler
class BaseSchedCB(Callback):
    order = TrainCB.order + 1

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
    order = 100

    def __init__(
        self, hookfunc, mod_filter=None, on_train=True, on_valid=False, mods=None
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
            mods = get_children(learn.model)
            if self.mod_filter:
                mods = filter(self.mod_filter, mods)
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
    act_filter = lambda m: isinstance(
        m, (nn.ReLU, nn.LeakyReLU, nn.GELU, GeneralReLU, nn.Sigmoid, nn.SiLU, Noop)
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
    def get_hist(h, lim=None):
        if not isinstance(lim, slice):
            lim = slice(lim)
        return torch.stack(h.stats[2][lim]).t().float().log1p()

    @staticmethod
    def get_min(h, lim=None):
        if not isinstance(lim, slice):
            lim = slice(lim)
        h1 = torch.stack(h.stats[2][lim]).t().float()
        return h1[0] / h1.sum(0)

    def color_dim(self, figsize=(11, 5), lim=None):
        fig, axes = get_grid(len(self), figsize=figsize)
        for ax, h in zip(axes.flat, self):
            show_image(self.get_hist(h, lim), ax, origin="lower")
        fig.show()

    def dead_chart(self, figsize=(11, 5), lim=None):
        fig, axes = get_grid(len(self), figsize=figsize)
        for ax, h in zip(axes.flatten(), self):
            ax.plot(self.get_min(h, lim=lim))
            ax.set_ylim(0, 1)
        fig.show()

    def plot_stats(self, figsize=(10, 4), lim=None):
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        if not isinstance(lim, slice):
            lim = slice(lim)
        for h in self:
            for i in 0, 1:
                axs[i].plot(h.stats[i][lim])
        axs[0].set_title("Means")
        axs[1].set_title("Stdevs")
        fig.legend(fc.L.range(self))
        fig.show()

    def plot_all(self, lim=None):
        self.color_dim(lim=lim)
        self.plot_stats(lim=lim)
        self.dead_chart(lim=lim)


## Accelerated Training


class AccelerateCB(TrainCB):
    order = TrainCB.order

    def __init__(self, n_inp=1, mixed_precision="fp16"):
        super().__init__(n_inp=n_inp)
        self.acc = Accelerator(mixed_precision=mixed_precision)

    def before_fit(self, learn):
        super().before_fit(learn)
        # require no other TrainCB, what about DeviceCB?
        learn.model, learn.opt, learn.dls.train, learn.dls.valid = self.acc.prepare(
            learn.model, learn.opt, learn.dls.train, learn.dls.valid
        )

    def backward(self, learn):
        self.acc.backward(learn.loss)


## Debug


class DebugTrain(Callback):
    order = 100

    def before_fit(self, learn):
        print("before_fit")

    def before_epoch(self, learn):
        print("before_epoch")

    def before_batch(self, learn):
        print("before_batch")

    def after_pred(self, learn):
        print("after_pred")

    def after_loss(self, learn):
        print(f"after_loss: {learn.loss}")

    def after_backward(self, learn):
        print("after_backward")

    def after_step(self, learn):
        print("after_step")

    def after_batch(self, learn):
        print("after_batch")

    def after_epoch(self, learn):
        print("after epoch")

    def after_fit(self, learn):
        print("after_fit")
