from __future__ import annotations
from functools import partial
import warnings
import fastcore.all as fc
from pathlib import Path
from datetime import datetime

import torch
from torch import optim
from torch.optim import lr_scheduler

from tinyai.core import MODEL_DIR, cls_name, def_device, IN_NOTEBOOK
from tinyai.cbs import CancelBatchException, CancelEpochException, CancelFitException
from tinyai.cbs import *
from tinyai.hooks import Hooks
from tinyai.viz import show_images

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
        self,
        model,
        dls,
        loss_func,
        lr=None,
        cbs=None,
        opt_func=partial(optim.AdamW, eps=1e-5),
        model_dir=MODEL_DIR,
    ):
        """
        AdamW eps 1e-5:
        When we divide by the exponential moving average of the squared gradient we
        don't want to divide by 0, So we add eps.
        If eps is really small, this can make the lr *huge*, so I increase it to 1e-5
        """
        self.model, self.dls, self.loss_func, self.lr, self.opt_func, self.model_dir = (
            model,
            dls,
            loss_func,
            lr,
            opt_func,
            Path(model_dir),
        )
        self.model_dir.mkdir(exist_ok=True, parents=False)
        self.cbs = fc.L(cbs)[:]

    def fit(
        self,
        nepochs,
        lr=None,
        cbs=None,
        ignore_cbs=None,
        train=True,
        valid=True,
    ):
        ## Settings for only this fit()
        new_cbs = fc.L(cbs)
        self.cbs.extend(new_cbs)
        self.ignore_cbs = ignore_cbs

        ## Setup
        self.train_steps = 0
        self.nepochs = nepochs
        self.epochs = range(nepochs)
        lr = lr or self.lr
        if lr is None:
            raise ValueError("Specify lr in either init or fit")

        self.opt = self.opt_func(self.model.parameters(), lr=lr)  # type: ignore

        ## Fit
        try:
            self._fit(train, valid)
        finally:
            if len(new_cbs):
                self.cbs = self.cbs[: -len(new_cbs)]
            self.ignore_cbs = None

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
            try:
                *_, h, w = outp.shape
            except AttributeError:
                print("Skipping:", cls_name(mod))
                return

            flops = sum(_flops(o, h, w) for o in mod.parameters()) / 1e6
            totf += flops
            res += f"|{cls_name(mod)}|{tuple(inp[0].shape)}|{tuple(outp.shape)}|{nparms}|{flops:.1f}|\n"

        with Hooks(list(self.model.modules()), _f) as hooks:
            self.fit(
                1,
                lr=1,
                train=False,
                cbs=NBatchCB(1),
                ignore_cbs=[ProgressCB, PlotCB],
            )
        print(f"Tot params: {totp}; MFLOPS: {totf:.1f}")
        if IN_NOTEBOOK:
            from IPython.display import Markdown

            return Markdown(res)
        else:
            print(res)

    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad_(False)

    def unfreeze(self):
        for p in self.model.parameters():
            p.requires_grad_(True)

    def save(self, fn=None, overwrite: bool = "warn"):  # type: ignore
        self.model_dir.mkdir(exist_ok=True, parents=True)

        if fn is None:
            fn = f"{datetime.now().strftime('%Y-%m-%d-%H:%M')}_{cls_name(self.model)}"

        save_file = self.model_dir / fn
        if save_file.exists():
            if overwrite == "warn":
                warnings.warn(f"{save_file} already exists, overwriting")
            elif not overwrite:
                warnings.warn(f"{save_file} already exists, exiting")
                return
        torch.save(self.model.state_dict(), save_file)

    def load(self, fn):
        self.model.load_state_dict(torch.load(self.model_dir / fn))


class Trainer(Learner):
    @fc.delegates(Learner.__init__)  # type: ignore
    def __init__(self, model, dls, loss_func, **kwargs):
        self.default_cbs = [
            DeviceCB(),
            AccelerateCB(n_inp=1) if def_device == "cuda" else TrainCB(n_inp=1),
            ProgressCB(),
            DefaultMetricsCB(),  # Only called if MetricsCB is not given at fit
        ]

        kwargs["cbs"] = self.default_cbs + fc.L(kwargs.get("cbs", []))
        super().__init__(model, dls, loss_func, **kwargs)

    def lr_find(self, gamma=1.3, max_mult=3, start_lr=1e-5, max_epochs=10):
        self.fit(
            max_epochs,
            lr=start_lr,
            cbs=LRFinderCB(gamma, max_mult),
            ignore_cbs=[PlotCB],
        )

    def validate(self, cbs=None):
        self.fit(1, lr=1000, train=False, valid=True, cbs=cbs, ignore_cbs=[PlotCB])

    @fc.delegates(Learner.fit)  # type: ignore
    def fit_one_cycle(self, nepochs, **kwargs):
        tmax = len(self.dls.train) * nepochs

        # repeated from fit, but scheduler must be passed by CB
        max_lr = kwargs.get("lr", None) or self.lr
        if max_lr is None:
            raise ValueError("Specify lr in either init or fit")

        sched = partial(lr_scheduler.OneCycleLR, max_lr=max_lr, total_steps=tmax)
        kwargs["cbs"] = [BatchSchedCB(sched)] + fc.L(kwargs.get("cbs", None))
        self.fit(nepochs, **kwargs)

    @fc.delegates(show_images)  # type: ignore
    def show_image_batch(self, max_n=9, cbs=None, **kwargs):
        self.fit(1, cbs=[NBatchCB(nbatches=1)] + fc.L(cbs))
        show_images(self.batch[0][:max_n], **kwargs)

    def capture_preds(self, cbs=None, inps=False):
        cp = CapturePreds()
        self.validate(cbs=[cp] + fc.L(cbs))
        res = cp.all_preds, cp.all_targs
        if inps:
            res = res + (cp.all_inps,)
        return res
