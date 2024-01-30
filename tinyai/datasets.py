from operator import itemgetter

import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, default_collate

__all__ = [
    "inplace",
    "transformi",
    "collate_dd",
    "DataLoaders",
    "BasicDataset",
    "get_dls",
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
    def __init__(self, train_ds, valid_ds):
        self.train, self.valid = train_ds, valid_ds

    @classmethod
    def from_dd(cls, dd, batch_size, num_workers=1):
        f = collate_dd(dd["train"])
        return cls(
            *[
                DataLoader(
                    ds, batch_size=batch_size, num_workers=num_workers, collate_fn=f
                )
                for ds in dd.values()
            ]
        )


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
