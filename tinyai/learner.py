from __future__ import annotations
import math
from operator import attrgetter
from functools import partial
import fastcore.all as fc
from fastprogress import progress_bar, master_bar
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from torcheval.metrics import MulticlassAccuracy, Mean

from tinyai.core import def_device, to_device, to_cpu


class CancelFitException(Exception): pass
class CancelBatchException(Exception): pass
class CancelEpochException(Exception): pass

class with_cbs:
    def __init__(self, nm): self.nm = nm
    def __call__(self, f):
        def _inner(o, *args, **kwargs):
            try:
                o.callback(f'before_{self.nm}')
                f(o, *args, **kwargs)
                o.callback(f'after_{self.nm}')
            except globals()[f'Cancel{self.nm.title()}Exception']: pass
            finally: o.callback(f'cleanup_{self.nm}')
        return _inner

def run_cbs(cbs, method_nm, learn=None, ignored=None):
    ignored = fc.L(ignored)
    for cb in sorted(cbs, key=attrgetter('order')):
        if type(cb).__name__ in ignored: continue
        method = getattr(cb, method_nm, None)
        if method is not None: method(learn)

class Learner:
    def __init__(self, model, dls, loss_func=F.mse_loss, lr=0.1, cbs=None, opt_func=optim.SGD):
        fc.store_attr()
        self.model = model.to(def_device)
        self.cbs = fc.L(cbs)

    def fit(self, n_epochs, train=True, valid=True, cbs=None, lr=None, ignore_cbs=None):
        self.ignore_cbs = fc.L(ignore_cbs)
        cbs = fc.L(cbs)
        for cb in cbs: self.cbs.append(cb)
        try:
            self.n_epochs = n_epochs
            self.epochs = range(n_epochs)
            if lr is None: lr = self.lr
            self.opt = self.opt_func(self.model.parameters(), lr=lr)
            self._fit(train, valid)
        finally:
            for cb in cbs: self.cbs.remove(cb)

    @with_cbs('fit')
    def _fit(self, train, valid):
        for self.epoch in self.epochs:
            if train: self.one_epoch(training=True)
            if valid: torch.no_grad()(self.one_epoch)(training=False)

    def one_epoch(self, training: bool):
        self.model.train(training)
        self.dl = self.dls.train if training else self.dls.valid
        self._one_epoch()

    @property
    def training(self): return self.model.training

    @with_cbs('epoch')
    def _one_epoch(self):
        for self.iter, self.batch in enumerate(self.dl): self._one_batch()

    @with_cbs('batch')
    def _one_batch(self):
        self.predict()
        self.callback('after_predict')
        self.get_loss()
        self.callback('after_loss')
        if self.training:
            self.backward()
            self.callback('after_backward')
            self.step()
            self.callback('after_step')
            self.zero_grad()

    def callback(self, method_nm): run_cbs(self.cbs, method_nm, learn=self, ignored=self.ignore_cbs)

    def __getattr__(self, name):
        if name in ('predict','get_loss','backward','step','zero_grad'): return partial(self.callback, name)
        raise AttributeError(name)

    def lr_find(self, gamma=1.3, max_mult=3, start_lr=1e-5, max_epochs=10):
        self.fit(max_epochs, lr=start_lr, cbs=LRFinderCB(gamma, max_mult))


class TrainLearner(Learner):
    @fc.delegates(Learner.__init__)
    def __init__(self, model, dls, **kwargs):
        kwargs['cbs'] = [ToDeviceCB(), TrainCB(n_inp=1), ProgressCB(plot=True)] + kwargs.get('cbs', [])
        super().__init__(model, dls, **kwargs)

### CALLBACKS ###

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
        for o in ms: metrics[type(o).__name__] = o
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
            log = dict(Epoch=learn.epoch, Train='train' if learn.model.training else 'eval', Loss=f"{self.loss.compute().item():.4f}",)
            log.update({k: f"{v.compute().item():.4f}" for k,v in self.metrics.items()})
            self._log(log)

class ProgressCB(Callback):
    order = 2
    def __init__(self, plot=False): self.plot = plot

    def before_fit(self, learn):
        learn.epochs = self.mbar = master_bar(learn.epochs)
        self.first = True
        if hasattr(learn, 'metrics'): learn.metrics._log = self._log
        self.losses = []
        self.val_losses = []

    def _log(self, d):
        if self.first:
            self.mbar.write(list(d), table=True)
            self.first = False
        self.mbar.write(list(d.values()), table=True)

    def before_epoch(self, learn):
        learn.dl = progress_bar(learn.dl, leave=False, parent=self.mbar)

    def after_batch(self, learn):
        learn.dl.comment = f'{learn.loss:.3f}'
        if self.plot and hasattr(learn, 'metrics') and learn.training:
            self.losses.append(learn.loss.item())
            if self.val_losses: self.mbar.update_graph([[fc.L.range(self.losses), self.losses],[fc.L.range(learn.epoch).map(lambda x: (x+1)*len(learn.dls.train)), self.val_losses]])

    def after_epoch(self, learn):
        if not learn.training:
            if self.plot and hasattr(learn, 'metrics'):
                self.val_losses.append(learn.metrics.loss.compute())
                self.mbar.update_graph([[fc.L.range(self.losses), self.losses],[fc.L.range(learn.epoch+1).map(lambda x: (x+1)*len(learn.dls.train)), self.val_losses]])


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
        self.best_metric = float('inf')

    def after_epoch(self, learn):
        if not learn.training:
            metric = self.metric.compute().item()
            if metric > self.best_metric:
                self.patience -= 1
                if self.patience == 0:
                    print(f"Early stopping {learn.epoch}: current {metric:.4f}, best {self.best_metric:.4f}")
                    raise CancelFitException()
            else:
                self.patience = self.init_patience
                self.best_metric = metric


class SingleBatchCB(Callback):
    order = 3
    def after_batch(self, learn): raise CancelFitException()


class LRFinderCB(Callback):
    order = 2
    def __init__(self, gamma=1.3, max_mult=3):
        fc.store_attr()
        super().__init__()

    def before_fit(self, learn):
        self.sched = ExponentialLR(learn.opt, gamma=self.gamma)
        self.lrs, self.losses = [], []
        self.min = float('inf')

    def before_epoch(self, learn):
        if not learn.training:
            raise CancelEpochException()

    def after_batch(self, learn):
        loss = to_cpu(learn.loss)
        self.lrs.append(learn.opt.param_groups[0]['lr'])
        self.losses.append(loss)
        if loss < self.min: self.min = loss
        if math.isnan(loss) or loss > self.min * self.max_mult:
            raise CancelFitException()
        self.sched.step()

    def cleanup_fit(self, learn):
        plt.plot(self.lrs, self.losses)
        plt.xscale('log')
