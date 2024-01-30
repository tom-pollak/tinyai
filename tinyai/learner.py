from __future__ import annotations
from functools import partial
from types import NoneType
import fastcore.all as fc

import torch
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler

from tinyai.core import cls_name, def_device
from tinyai.cbs import CancelBatchException, CancelEpochException, CancelFitException
from tinyai.cbs import *
from tinyai.hooks import Hooks

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
        self, model, dls, loss_func, lr=None, cbs=None, opt_func=optim.SGD
    ):
        self.model, self.dls, self.loss_func, self.lr, self.opt_func = (
            model,
            dls,
            loss_func,
            lr,
            opt_func,
        )
        self.cbs = fc.L(cbs)

    def fit(
        self,
        nepochs,
        train=True,
        valid=True,
        cbs=None,
        lr=None,
        ignore_cbs=None,
    ):
        ## Settings for only this fit()
        self.ignore_cbs = ignore_cbs
        self.cbs.extend(fc.L(cbs))

        ## Setup
        self.train_steps = 0
        self.nepochs = nepochs
        self.epochs = range(nepochs)
        lr = lr or self.lr
        if lr is None:
            raise ValueError("Specify lr in either init or fit")

        self.model = self.model.to(def_device)
        self.opt = self.opt_func(self.model.parameters(), lr=lr)  # type: ignore

        ## Fit
        try:
            self._fit(train, valid)
        finally:
            self.ignore_cbs = None
            for cb in fc.L(cbs):
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
        if self.training:
            self.train_steps += 1
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

    def summary(self):
        def _flops(x, h, w) -> int:  # type: ignore
            if x.dim() < 3:
                return x.numel()
            if x.dim() == 4:
                return x.numel() * h * w

        res = "|Module|Input|Output|Num params|MFLOPS|\n|--|--|--|--|--|\n"
        totp, totf = 0, 0

        def _f(hook, mod, inp, outp):
            nonlocal res, totp, totf
            nparms = sum(o.numel() for o in mod.parameters())
            totp += nparms
            *_, h, w = outp.shape
            flops = sum(_flops(o, h, w) for o in mod.parameters()) / 1e6
            totf += flops
            res += f"|{cls_name(mod)}|{tuple(inp[0].shape)}|{tuple(outp.shape)}|{nparms}|{flops:.1f}|\n"

        with Hooks(self.model, _f) as hooks:
            self.fit(
                1,
                lr=1,
                train=False,
                cbs=NBatchCB(),
                ignore_cbs=[ProgressCB, PlotCB],
            )
        print(f"Tot params: {totp}; MFLOPS: {totf:.1f}")
        if fc.IN_NOTEBOOK:
            from IPython.display import Markdown

            return Markdown(res)
        else:
            print(res)


class Trainer(Learner):
    default_cbs = [
        ToDeviceCB(),
        TrainCB(n_inp=1),
        MetricsCB(),
        ProgressCB(),
        PlotLossCB(),
    ]

    @fc.delegates(Learner.__init__)  # type: ignore
    def __init__(self, model, dls, **kwargs):
        kwargs["cbs"] = self.default_cbs + kwargs.get("cbs", [])
        super().__init__(model, dls, **kwargs)

    @fc.delegates(Learner.fit)  # type: ignore
    def fit(self, nepochs, metrics=None, **kwargs):
        # Add metrics
        if isinstance(metrics, (list, NoneType)):
            metrics = {cls_name(m): m for m in fc.L(metrics)}
        self.default_cbs[2].metrics.update(**metrics)

        super().fit(nepochs, **kwargs)

    def lr_find(self, gamma=1.3, max_mult=3, start_lr=1e-5, max_epochs=10):
        self.fit(
            max_epochs,
            lr=start_lr,
            cbs=LRFinderCB(gamma, max_mult),
            ignore_cbs=[PlotCB],
        )

    def validate(self):
        self.fit(1, train=False, valid=True, ignore_cbs=[PlotCB])

    fc.delegates(Learner.fit)  # type: ignore

    def fit_one_cycle(self, nepochs, max_lr, **kwargs):
        if "lr" in kwargs:
            raise ValueError("fit_one_cycle uses max_lr, do not set lr")

        tmax = len(self.dls.train) * nepochs
        sched = partial(lr_scheduler.OneCycleLR, max_lr=max_lr, total_steps=tmax)
        kwargs["cbs"] = [BatchSchedCB(sched)] + fc.L(kwargs.get("cbs", None))
        self.fit(nepochs, **kwargs)
