from functools import partial


__all__ = ["Hook", "Hooks"]


class Hook:
    def __init__(self, m, f, forward=True):
        self.m = m
        if forward:
            self.hook = m.register_forward_hook(partial(f, self))
        else:
            self.hook = m.register_full_backward_hook(partial(f, self))

    def remove(self):
        self.m = None
        self.hook.remove()

    def __del__(self):
        self.remove()

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()


class Hooks(list):
    def __init__(self, ms, f, forward=True):
        super().__init__([Hook(m, f, forward) for m in ms])

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()

    def __del__(self):
        self.remove()

    def __delitem__(self, i):
        self[i].remove()
        super().__delitem__(i)

    def remove(self):
        for h in self:
            h.remove()
