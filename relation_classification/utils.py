# -*- coding:utf8 -*-

import os

__all__ = ['G', 'g', 'AverageMeter', 'load_source', 'map_exec']


class G(dict):
    def __getattr__(self, k):
        if k not in self:
            raise AttributeError
        return self[k]

    def __setattr__(self, k, v):
        self[k] = v

    def __delattr__(self, k):
        del self[k]

    def print(self, sep=': ', end='\n', file=None):
        keys = sorted(self.keys())
        lens = list(map(len, keys))
        max_len = max(lens)
        for k in keys:
            print(k + ' ' * (max_len - len(k)), self[k], sep=sep, end=end, file=file, flush=True)

g = G()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    val = 0
    avg = 0
    sum = 0
    count = 0

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_source(filename, name=None):
    import imp

    if name is None:
        basename = os.path.basename(filename)
        if basename.endswith('.py'):
            basename = basename[:-3]
        name = basename.replace('.', '_')

    return imp.load_source(name, filename)


def map_exec(func, iterable, ret_type=list):
    return ret_type(map(func, iterable))

