from __future__ import annotations
import math
import fastcore.all as fc
from operator import attrgetter
import matplotlib.pyplot as plt
from fastprogress import progress_bar, master_bar

from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import default_collate
from torcheval.metrics import Mean


from tinyai.core import def_device, to_device, to_cpu
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
    "EarlyStoppingCB",
    "SingleBatchCB",
    "LRFinderCB",
    "BatchTransformCB",
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
        if type(cb).__name__ in ignored:
            continue
        method = getattr(cb, method_nm, None)
        if method is not None:
            method(learn)


class Callback:
    order = 0


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
    show_train = False

    def __init__(self, *ms, **metrics):
        for o in ms:
            metrics[type(o).__name__] = o
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
        if self.show_train or not learn.training:
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

    def __init__(self, plot=False):
        self.init_plot = plot

    def before_fit(self, learn):
        learn.epochs = self.mbar = master_bar(learn.epochs)
        self.first = True
        if hasattr(learn, "metrics"):
            learn.metrics._log = self._log
        self.losses = []
        self.val_losses = []

    def _log(self, d):
        if self.first:
            self.mbar.write(list(d), table=True)
            self.first = False
        self.mbar.write(list(d.values()), table=True)

    def before_epoch(self, learn):
        learn.dl = progress_bar(learn.dl, leave=False, parent=self.mbar)
        self.plot = (
            self.init_plot  # cb allows plot
            and hasattr(learn, "metrics")  # metrics to plot
            and (learn.plot if learn.plot is not None else True)  # user allows plot
        )

    def _graph_data(self):
        return [
            [fc.L.range(self.losses), self.losses],
            [x.numpy() for x in default_collate(self.val_losses)]
            if len(self.val_losses)
            else [],
        ]

    def after_batch(self, learn):
        learn.dl.comment = f"{learn.loss:.3f}"
        if self.plot and learn.training:
            self.losses.append(learn.loss.item())
            self.mbar.update_graph(self._graph_data())

    def after_epoch(self, learn):
        if not learn.training:
            if self.plot:
                self.val_losses.append((len(self.losses), learn.metrics.loss.compute()))
                self.mbar.update_graph(self._graph_data())


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

    def __init__(self, eval_nepochs=float("inf")):
        self.eval_nepochs = eval_nepochs

    def after_batch(self, learn):
        # cancel after one training batch
        if learn.training:
            raise CancelEpochException()

    def before_batch(self, learn):
        # if validating, and not eval epoch, cancel validation
        if not learn.training and (learn.epoch + 1) % self.eval_nepochs != 0:
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
        learn.model.load_state_dict(self.initial_state)

        plt.plot(self.lrs, self.losses)
        plt.xscale("log")


class BatchTransformCB(Callback):
    def __init__(self, tfm):
        self.tfm = tfm

    def before_batch(self, learn):
        learn.batch = self.tfm(learn.batch)
