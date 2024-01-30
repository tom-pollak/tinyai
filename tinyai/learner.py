from __future__ import annotations
from functools import partial
import fastcore.all as fc

import torch
import torch.nn.functional as F
from torch import optim

from tinyai.core import def_device
from tinyai.cbs import CancelBatchException, CancelEpochException, CancelFitException
from tinyai.cbs import *

__all__ = [
    "with_cbs",
    "Learner",
    "Trainer",
]


class with_cbs:
    def __init__(self, nm):
        self.nm = nm

    def __call__(self, f):
        def _inner(o, *args, **kwargs):
            try:
                o.callback(f"before_{self.nm}")
                f(o, *args, **kwargs)
                o.callback(f"after_{self.nm}")
            except globals()[f"Cancel{self.nm.title()}Exception"]:
                pass
            finally:
                o.callback(f"cleanup_{self.nm}")

        return _inner


class Learner:
    def __init__(
        self, model, dls, loss_func=F.mse_loss, lr=0.1, cbs=None, opt_func=optim.SGD
    ):
        self.model, self.dls, self.loss_func, self.lr, self.opt_func = (
            model,
            dls,
            loss_func,
            lr,
            opt_func,
        )
        self.cbs = fc.L(cbs)
        self.plot = None

    def fit(
        self,
        n_epochs,
        train=True,
        valid=True,
        cbs=None,
        lr=None,
        ignore_cbs=None,
        plot=None,
    ):
        ## Settings for only this fit()
        if plot is not None:
            self.plot = plot
        self.ignore_cbs = fc.L(ignore_cbs)
        cbs = fc.L(cbs)
        for cb in cbs:
            self.cbs.append(cb)

        ## Setup
        self.model = self.model.to(def_device)
        self.n_epochs = n_epochs
        self.epochs = range(n_epochs)
        if lr is None:
            lr = self.lr
        self.opt = self.opt_func(self.model.parameters(), lr=lr)  # type: ignore

        try:
            self._fit(train, valid)
        finally:
            ## Remove settings
            self.plot = None
            for cb in cbs:
                self.cbs.remove(cb)

    @with_cbs("fit")
    def _fit(self, train, valid):
        for self.epoch in self.epochs:
            if train:
                self.one_epoch(training=True)
            if valid:
                torch.no_grad()(self.one_epoch)(training=False)

    def one_epoch(self, training: bool):
        self.model.train(training)
        self.dl = self.dls.train if training else self.dls.valid  # type: ignore
        self._one_epoch()

    @property
    def training(self):
        return self.model.training

    @with_cbs("epoch")
    def _one_epoch(self):
        for self.iter, self.batch in enumerate(self.dl):
            self._one_batch()

    @with_cbs("batch")
    def _one_batch(self):
        self.predict()
        self.callback("after_predict")
        self.get_loss()
        self.callback("after_loss")
        if self.training:
            self.backward()
            self.callback("after_backward")
            self.step()
            self.callback("after_step")
            self.zero_grad()

    def callback(self, method_nm):
        run_cbs(self.cbs, method_nm, learn=self, ignored=self.ignore_cbs)

    def __getattr__(self, name):
        if name in ("predict", "get_loss", "backward", "step", "zero_grad"):
            return partial(self.callback, name)
        raise AttributeError(name)


class Trainer(Learner):
    @fc.delegates(Learner.__init__)  # type: ignore
    def __init__(self, model, dls, **kwargs):
        kwargs["cbs"] = [
            ToDeviceCB(),
            TrainCB(n_inp=1),
            ProgressCB(plot=True),
        ] + kwargs.get("cbs", [])
        super().__init__(model, dls, **kwargs)

    def lr_find(self, gamma=1.3, max_mult=3, start_lr=1e-5, max_epochs=10):
        self.fit(max_epochs, lr=start_lr, cbs=LRFinderCB(gamma, max_mult), plot=False)

    def validate(self):
        self.fit(1, train=False, valid=True, plot=False)
