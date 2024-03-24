import math
from itertools import zip_longest

import fastcore.all as fc
import matplotlib.pyplot as plt
import numpy as np
import torch

__all__ = ["show_image", "subplots", "get_grid", "show_images", "plot_hist", "plot_confusion_matrix"]


@fc.delegates(plt.Axes.imshow)  # type: ignore
def show_image(im, ax=None, figsize=None, title=None, noframe=True, **kwargs):
    "Show a PIL or PyTorch image on `ax`."
    if im is None:
        return ax
    if fc.hasattrs(im, ("cpu", "permute", "detach")):
        im = im.detach().cpu()
        if len(im.shape) == 3 and im.shape[0] < 5:
            im = im.permute(1, 2, 0)
        im = im.numpy()
    elif not isinstance(im, np.ndarray):
        im = np.array(im)
    if im.shape[-1] == 1:
        im = im[..., 0]
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.imshow(im, **kwargs)
    if title is not None:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if noframe:
        ax.axis("off")
    return ax


@fc.delegates(plt.subplots, keep=True)  # type: ignore
def subplots(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple | None = None,
    imsize: int = 3,  # size of each image
    suptitle: str | None = None,
    **kwargs,
):  # fig and axs
    "A figure and set of subplots to display images of `imsize` inches"
    if figsize is None:
        figsize = (ncols * imsize, nrows * imsize)
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    if suptitle is not None:
        fig.suptitle(suptitle)
    if nrows * ncols == 1:
        ax = np.array([ax])
    return fig, ax


@fc.delegates(subplots)  # type: ignore
def get_grid(
    n: int,
    nrows: int | None = None,
    ncols: int | None = None,
    title: str | None = None,
    weight: str = "bold",
    size: int = 14,
    **kwargs,
):  # fig and axs
    "Return a grid of `n` axes, `rows` by `cols`"
    if nrows:
        ncols = ncols or int(math.ceil(n / nrows))
    elif ncols:
        nrows = nrows or int(math.ceil(n / ncols))
    else:
        ncols = int(math.ceil(math.sqrt(n)))
        nrows = int(math.ceil(n / ncols))
    fig, axs = subplots(nrows, ncols, **kwargs)
    for i in range(n, nrows * ncols):
        axs.flat[i].set_axis_off()
    if title is not None:
        fig.suptitle(title, weight=weight, size=size)
    return fig, axs


@fc.delegates(subplots)  # type: ignore
def show_images(
    ims: list,
    nrows: int | None = None,
    ncols: int | None = None,
    titles: list | None = None,
    **kwargs,
):
    "Show all images `ims` as subplots with `rows` using `titles`"
    axs = get_grid(len(ims), nrows, ncols, **kwargs)[1].flat
    for im, t, ax in zip_longest(ims, titles or [], axs):
        show_image(im, ax=ax, title=t)
    plt.tight_layout()


def plot_hist(x, ax=None, range=None):
    hist, bins = torch.histogram(x, range=range)
    bin_widths = bins[1:] - bins[:-1]
    bin_centers = bins[:-1] + bin_widths / 2

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))

    ax.bar(bin_centers, hist, width=bin_widths, align="center")
    ax.set_ylabel("Frequency")


import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(y_preds, y_true, labels=None):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(12, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", ax=ax, values_format=".2f", colorbar=False)
    plt.title("Normalized Confusion Matrix")
    ax.tick_params(axis="x", labelrotation=90)

