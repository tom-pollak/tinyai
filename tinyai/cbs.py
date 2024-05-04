from __future__ import annotations
from copy import deepcopy
import os
from functools import partial
import math
import fastcore.all as fc
from operator import attrgetter
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from IPython.display import display, DisplayHandle
import pandas as pd

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import default_collate
from torcheval.metrics import Mean
from accelerate import Accelerator


from tinyai.core import (
    IN_NOTEBOOK,
    cls_name,
    def_device,
    get_children,
    to_device,
    to_cpu,
    Noop,
)
from tinyai.hooks import Hooks
from tinyai._nn.act import GeneralReLU
from tinyai.viz import show_image, get_grid

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

tqdm = tqdm_notebook if IN_NOTEBOOK else tqdm


__all__ = [
    "CancelFitException",
    "CancelBatchException",
    "CancelEpochException",
    "run_cbs",
    "Callback",
    "DeviceCB",
    "BaseTrainCB",
    "TrainCB",
    "MetricsCB",
    "DefaultMetricsCB",
    "ProgressCB",
    "PlotCB",
    "PlotLossCB",
    "PlotMetricsCB",
    "EarlyStoppingCB",
    "CheckpointCB",
    "OptunaPruningCB",
    "NBatchCB",
    "OverfitBatchCB",
    "LRFinderCB",
    "BatchTransformCB",
    "BaseSchedCB",
    "BatchSchedCB",
    "EpochSchedCB",
    "RecorderCB",
    "HooksCallback",
    "ActivationStats",
    "GradStats",
    "AccelerateCB",
    "CapturePreds",
    "DistillTrainCB",
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
        raise ValueError(f"Required callback {required_cbs}", cbs)


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
        x, *_ = learn.batch
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
    required_cbs = [BaseTrainCB]

    def __init__(self, sigmoid=False, show_train=True, do_log=True, **metrics):
        self.sigmoid, self.show_train, self.do_log = sigmoid, show_train, do_log
        self.metrics = metrics

    def _create_log(self, learn):
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
                        {f"{k}_{i}": f"{o.item():.4f}" for i, o in enumerate(metric)}
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
        return log

    def _log(self, learn, log):
        first = log["epoch"] == 0 and (log["train"] == "train" or not self.show_train)
        row_df = pd.DataFrame.from_dict({k: [v] for k, v in log.items()})
        row_str = row_df.to_string(
            index=False,
            justify="right",
            formatters={col: "{: >8}".format for col in row_df.columns},
        )
        if not first:
            row_str = row_str.split("\n")[1]
        print(row_str)
        # html_args = dict(index=False, header=first, notebook=IN_NOTEBOOK)
        # display(HTML(row_df.to_html(**html_args)))  # type: ignore

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
        if self.do_log and (learn.training and self.show_train) or self.do_log and not learn.training:
            log = self._create_log(learn)
            self._log(learn, log)


class DefaultMetricsCB(MetricsCB):
    def before_fit(self, learn):
        self.enabled = not any(
            isinstance(cb, MetricsCB) and not isinstance(cb, DefaultMetricsCB)
            for cb in learn.cbs
        )
        if self.enabled:
            super().before_fit(learn)

    def before_epoch(self, learn):
        if self.enabled:
            return super().before_epoch(learn)

    def after_batch(self, learn):
        if self.enabled:
            return super().after_batch(learn)

    def after_epoch(self, learn):
        if self.enabled:
            return super().after_epoch(learn)


class ProgressCB(Callback):
    order = MetricsCB.order + 1

    def before_fit(self, learn):
        learn.epochs = self.mbar = tqdm(learn.epochs, desc="Epoch", leave=True)
        self.pbar = tqdm(total=0, leave=True, desc="Batch")

    def before_epoch(self, learn):
        self.pbar.reset(total=len(learn.dl))

    def after_batch(self, learn):
        self.pbar.set_postfix(loss=f"{learn.loss:.3f}")
        self.pbar.update(1)

    def cleanup_fit(self, learn):
        self.mbar.close()
        if hasattr(self, "pbar"):
            self.pbar.close()
            del self.pbar


class PlotCB(Callback):
    order = ProgressCB.order + 1

    def __init__(
        self,
        plot_every=10,
        train=True,
        valid=True,
        names=("train", "valid"),
        figsize=(6, 4),
        graph_kwargs=None,
        plot_kwargs=None,
    ):
        super().__init__()
        self.graph_kwargs = graph_kwargs or {}
        self.plot_kwargs = plot_kwargs or {}
        self.plot_every = plot_every
        self.train, self.valid = train, valid
        self.names = names
        self.figsize = figsize

    # TODO: only graph kwargs accepted are x_bounds, y_bounds
    def update_graph(self, graphs, x_bounds=None, y_bounds=None, **plot_kwargs):
        if not hasattr(self, "graph_fig"):
            self.graph_fig, self.graph_ax = plt.subplots(1, figsize=self.figsize)
            self.graph_out = display(self.graph_ax.figure, display_id=True)  # type: ignore
            assert isinstance(self.graph_out, DisplayHandle)
        self.graph_out: DisplayHandle

        self.graph_ax.clear()
        if len(self.names) < len(graphs):
            self.names += [""] * (len(graphs) - len(self.names))
        for g, n in zip(graphs, self.names):
            self.graph_ax.plot(*g, label=n, **plot_kwargs)
        if len(graphs):
            self.graph_ax.legend(loc="upper right")
        if x_bounds is not None:
            self.graph_ax.set_xlim(*x_bounds)
        if y_bounds is not None:
            self.graph_ax.set_ylim(*y_bounds)
        self.graph_out.update(self.graph_ax.figure)

    def after_batch(self, learn):
        if (
            self.train and learn.training and learn.iter % self.plot_every == 0
        ):  # store train
            self.update_graph(self._graph_data(), **self.graph_kwargs, **self.plot_kwargs)  # type: ignore

    def before_fit(self, learn):
        # create graph before logging
        self.update_graph([], **self.graph_kwargs, **self.plot_kwargs)  # type: ignore

    def after_epoch(self, learn):
        if self.valid and not learn.training:  # store valid
            self.update_graph(self._graph_data(), **self.graph_kwargs, **self.plot_kwargs)  # type: ignore

    def _graph_data(self):
        return []

    def after_fit(self, learn):
        plt.close(self.graph_fig)

    def cleanup_fit(self, learn):
        if hasattr(self, "graph_fig"):
            plt.close(self.graph_fig)
            del self.graph_fig, self.graph_ax, self.graph_out


class PlotLossCB(PlotCB):
    order = PlotCB.order + 1

    def __init__(
        self,
        train=True,
        valid=True,
        plot_every: int = float("inf"),
        graph_kwargs=None,
        plot_kwargs=None,
    ):
        super().__init__(
            plot_every=plot_every,
            train=train,
            valid=valid,
            graph_kwargs=graph_kwargs,
            plot_kwargs=plot_kwargs,
        )
        self.required_cbs = [BaseTrainCB]

    def before_fit(self, learn):
        require_cbs(learn.cbs, self.required_cbs)
        self.train_losses = []
        self.val_losses = []
        super().before_fit(learn)

    def after_batch(self, learn):
        if self.train and learn.training:  # store train
            self.train_losses.append((learn.train_steps, to_cpu(learn.loss)))
        super().after_batch(learn)

    def after_epoch(self, learn):
        if self.valid and not learn.training:  # store valid
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
    order = PlotLossCB.order + 1

    def __init__(self):
        super().__init__(graph_kwargs=dict(x_bounds=(0, None)))
        self.required_cbs = [MetricsCB]

    def before_fit(self, learn):
        require_cbs(learn.cbs, self.required_cbs)
        self.recs = {k: [] for k in learn.metrics.metrics.keys()}
        super().before_fit(learn)

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


import optuna


class OptunaPruningCB(MetricsCB):

    order = 100

    def __init__(self, trial, metric=None):
        "metric = None uses val loss"
        self.trial = trial
        if metric is not None:
            super().__init__(es_metric=metric)
            self.metric = self.metrics["es_metric"]

    def before_fit(self, learn):
        if self.metric is None:
            self.metric = learn.epoch_loss

    def after_epoch(self, learn):
        if not learn.training:
            metric = self.metric.compute().item()
            self.trial.report(metric, learn.epoch)
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()


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

    def __init__(self, nbatches=1, train=True, valid=False):
        self.nbatches = nbatches
        self.train, self.valid = train, valid

    def after_batch(self, learn):
        if learn.iter + 1 >= self.nbatches and (
            (self.train and learn.training) or (self.valid and not learn.training)
        ):
            raise CancelEpochException()


class OverfitBatchCB(Callback):
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

        fig, ax = plt.subplots()
        ax.plot(self.lrs, self.losses)
        ax.set_xscale("log")
        fig.show()


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
            fig, ax = plt.subplots()
            ax.plot(v, label=k)
            fig.show()


##
class HooksCallback(Callback):
    order = 100

    def __init__(
        self,
        hookfunc,
        mod_filter=None,
        on_train=True,
        on_valid=False,
        mods=None,
        forward=True,
    ):
        (
            self.hookfunc,
            self.mod_filter,
            self.on_train,
            self.on_valid,
            self.mods,
            self.forward,
        ) = (hookfunc, mod_filter, on_train, on_valid, mods, forward)
        super().__init__()

    def before_fit(self, learn):
        if self.mods:
            mods = self.mods
        else:
            mods = get_children(learn.model)
            if self.mod_filter:
                mods = filter(self.mod_filter, mods)
        self.hooks = Hooks(mods, partial(self._hookfunc, learn), forward=self.forward)

    def _hookfunc(self, learn, *args, **kwargs):
        if (self.on_train and learn.training) or (self.on_valid and not learn.training):
            self.hookfunc(*args, **kwargs)

    def after_fit(self, learn):
        self.hooks.remove()

    def __iter__(self):
        return iter(self.hooks)

    def __len__(self):
        return len(self.hooks)


class GradStats(HooksCallback):
    act_filter = lambda m: isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d))

    def __init__(self, mod_filter=act_filter):
        super().__init__(
            self.append_grad_stats,
            mod_filter,
            on_train=True,
            on_valid=False,
            forward=False,
        )

    @staticmethod
    def append_grad_stats(hook, mod, grad_inps, grad_outps):
        if not hasattr(hook, "grad_stats"):
            hook.grad_stats = ([], [])

        inp_norm = torch.stack([g.norm(p=2) for g in grad_inps]).norm(p=2).cpu()
        outp_norm = torch.stack([g.norm(p=2) for g in grad_outps]).norm(p=2).cpu()
        hook.grad_stats[0].append(inp_norm)
        hook.grad_stats[1].append(outp_norm)

    def plot_stats(self, figsize=(10, 4), lim=None):
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        if not isinstance(lim, slice):
            lim = slice(lim)
        for h in self:
            for i in 0, 1:
                axs[i].plot(h.grad_stats[i][lim])

        axs[0].set_title("Input Grad Norm")
        axs[1].set_title("Output Grad Norm")
        fig.legend(fc.L.range(self))
        fig.show()

    def plot_all(self, lim=None):
        self.plot_stats(lim=lim)


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
        acts = to_cpu(outp).float()
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


class CapturePreds(Callback):
    def before_fit(self, learn):
        self.all_inps, self.all_preds, self.all_targs = [], [], []

    def after_batch(self, learn):
        if hasattr(learn, "n_inp"):
            n_inp = learn.n_inp
        else:
            n_inp = 1

        self.all_preds.append(to_cpu(learn.preds))
        self.all_inps.append(to_cpu(learn.batch[0]))
        self.all_targs.append(to_cpu(learn.batch[1]))

    def after_fit(self, learn):
        self.all_preds, self.all_targs, self.all_inps = map(
            torch.cat, [self.all_preds, self.all_targs, self.all_inps]
        )


# ---


class DistillTrainCB(TrainCB):
    def __init__(self, teacher_model, temperature=2.0, alpha=0.5):
        super().__init__(n_inp=1)
        self.teacher_model = teacher_model
        self.loss_func = nn.KLDivLoss(reduction="batchmean")
        self.temperature, self.alpha = temperature, alpha

    def predict(self, learn):
        super().predict(learn)  # compute learn.preds
        with torch.inference_mode():
            self.teacher_preds = self.teacher_model(*learn.batch[: self.n_inp])

    def get_loss(self, learn):
        super().get_loss(learn)  # compute learn.loss

        loss_kd = self.temperature**2 * self.loss_func(
            F.log_softmax(learn.preds / self.temperature, dim=-1),
            F.softmax(self.teacher_preds / self.temperature, dim=-1),
        )
        learn.loss = self.alpha * learn.loss + (1.0 - self.alpha) * loss_kd
