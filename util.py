import torch
import random
import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class Normalizer():
    def __init__(self, input_shape, max_val=None):
        self.n = 0
        self.mean = torch.zeros(input_shape)
        self.mean_diff = torch.zeros(input_shape)
        self.var = torch.zeros(input_shape)
        self.max_val = max_val

    def observe(self, x):
        if self.max_val:
            return

        x = x.view(-1)
        self.n += 1.
        last_mean = self.mean.clone()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = torch.clamp(self.mean_diff / self.n, min=1e-2)

    def normalize(self, inputs):
        if self.max_val:
            return inputs / self.max_val

        obs_std = torch.sqrt(self.var)
        return (inputs - self.mean) / obs_std

    def denormalize(self, inputs):
        if self.max_val:
            return inputs * self.max_val

        obs_std = torch.sqrt(self.var)
        return (inputs * obs_std) + self.mean


class Log(object):
    def __init__(self):
        self.storage = {}

    def add(self, name, value):
        try:
            self.storage[name].append(value)
        except KeyError:
            self.storage[name] = [value]

    def get(self):
        return self.storage


def meanSmoothing(x, N):
    x = np.array(x)
    out = np.zeros_like(x, dtype=np.float64)
    dim_len = x.shape[0]
    for i in range(dim_len):
        if N % 2 == 0:
            a, b = i - (N - 1) // 2, i + (N - 1) // 2 + 2
        else:
            a, b = i - (N - 1) // 2, i + (N - 1) // 2 + 1
        # cap indices to min and max indices
        a = max(0, a)
        b = min(dim_len, b)
        out[i] = np.mean(x[a:b])
    return out


def display_top_memory_users(key_type='lineno', limit=3, censored=True):
    snapshot = tracemalloc.take_snapshot()
    if censored:
        snapshot = snapshot.filter_traces((
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        ))
        print("Top %s memory usage lines" % limit)
    else:
        limit = 0
    top_stats = snapshot.statistics(key_type)

    print(("Unc" if not censored else "C") + "ensored memory usage:")
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB" % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)
    other = top_stats[limit:]
    if other and censored:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))
    print()
