from operator import itemgetter

import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, default_collate

from tinyai.core import def_workers

__all__ = [
    "inplace",
    "transformi",
    "collate_dd",
    "DataLoaders",
    "BasicDataset",
    "get_dls",
    "MultDL",
    "LengthDataset",
    "get_dummy_dls",
]


def inplace(f):
    def _inner(x):
        f(x)
        return x

    return _inner


@inplace
def transformi(b, x="image"):
    b[x] = [TF.to_tensor(o) for o in b[x]]


def collate_dd(ds):
    get = itemgetter(*ds.features)

    def _inner(b):
        return get(default_collate(b))

    return _inner


class DataLoaders:
    def __init__(self, train_ds, valid_ds, test_ds=None):
        self.train, self.valid, self.test = train_ds, valid_ds, test_ds

    @classmethod
    def from_dd(
        cls,
        dd,
        batch_size,
        num_workers=def_workers,
        pin_memory=True,
        collate_trn=None,
        collate_tst=None,
        sampler=None,
        **kwargs,
    ):
        if collate_trn is None and collate_tst is None:
            collate_trn = collate_tst = collate_dd(dd["train"])

        def _create_dl(split, ds):
            _shuffle = False
            _sampler = None
            _drop_last = False
            if split == "train":
                collate_fn = collate_trn
                bs = batch_size
                _drop_last = True
                if sampler is not None:
                    _sampler = sampler
                else:
                    _shuffle = True
            elif split in {"validation", "test"}:
                collate_fn = collate_tst
                bs = batch_size * 2
            else:
                raise ValueError(f"Unknown split: {split}")

            return DataLoader(
                ds,
                batch_size=bs,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                shuffle=_shuffle,
                sampler=_sampler,
                drop_last=_drop_last,
                **kwargs,
            )

        return cls(*[_create_dl(split, ds) for split, ds in dd.items()])


#####


class BasicDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x)


def get_dls(train_ds, valid_ds, bs, **kwargs):
    # valid_dl 2*bs as no backprop, so should be able to fit 2x in memory
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
        DataLoader(valid_ds, batch_size=bs * 2, **kwargs),
    )


class MultDL:
    def __init__(self, dl, mult=2):
        self.dl, self.mult = dl, mult

    def __len__(self):
        return len(self.dl) * self.mult

    def __iter__(self):
        for o in self.dl:
            for i in range(self.mult):
                yield o


class LengthDataset(Dataset):
    def __init__(self, length=1):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return 0, 0


def get_dummy_dls(length=100):
    return DataLoaders(
        DataLoader(LengthDataset(length), batch_size=1),
        DataLoader(LengthDataset(length=1), batch_size=1),
    )
