import torch
import random
import numpy as np
import os
import sys
import logging
import matplotlib
matplotlib.use('Pdf')
import matplotlib.pyplot as plt

from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program


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
    def __init__(self, path, log, comment, log_NNs):
        self.do_logging = log
        print("Log gradients and weights: ", self.do_logging)

        self.log_NNs = log_NNs
        self.comment = comment
        self.writer = SummaryWriter(comment=comment)
        self.storage = {}
        self.global_step = 0
        self.tb_path = 'runs'
        self.run_tb()



    def run_tb(self):
        if not self.do_logging:
            return

        logging.getLogger('tensorflow').disabled = True
        logging.getLogger('werkzeug').setLevel(logging.ERROR)

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        logging.getLogger('tensorflow').setLevel(logging.FATAL)

        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.tb_path])
        url = tb.launch()

        sys.stdout.write('TensorBoard at %s \n' % url)
        sys.stdout.write('TensorBoard log dir %s\n' % self.tb_path)

    def add(self, name, value, distribution=None, steps=None):
        if not self.do_logging:
            return

        if steps is None:
            steps = self.global_step
        # Add to tensorboard:
        if distribution is None:
            self.writer.add_scalar(name, value, global_step=steps)
        else:
            self.writer.add_histogram(name, distribution, global_step=steps)

        try:
            self.storage[name].append(value)
        except KeyError:
            self.storage[name] = [value]

    def get(self):
        return self.storage

    def step(self):
        self.global_step += 1

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

def calc_gradient_norm(layers):
    total_norm = 0
    for p in layers.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** (1. / 2)

def calc_norm(layers):
    total_norm = torch.tensor(0.)
    for param in layers.parameters():
        total_norm += torch.norm(param)
    return total_norm

def filter_weight_dict(weight_dict):
    return {k: v for k, v in weight_dict.items()
            if "target_net" not in k and
            "F_s" not in k and
            "Q" not in k and
            "V" not in k}

def calculate_reduced_idxs(len_of_point_list, max_points):
    if max_points != 0:
        step_size = len_of_point_list // max_points
        step_size += 1 if len_of_point_list % max_points else 0
    else:
        return range(len_of_point_list)
    return range(0, len_of_point_list, step_size)


def reducePoints(list_of_points, max_points_per_line):
    if max_points_per_line != 0:
        step_size = len(list_of_points) // max_points_per_line
        step_size += 1 if len(list_of_points) % max_points_per_line else 0
    else:
        return range(len(list_of_points)), list_of_points
    steps = range(0, len(list_of_points), step_size)
    list_of_points = [np.mean(list_of_points[i:i + step_size]) for i in steps]
    return list_of_points


def mean_final_percent(result_list, percentage=0.1):
    final_percent_idx = int(len(result_list) * (1 - percentage))
    return np.mean(result_list[final_percent_idx:])


def run_metric(result_list, percentage=0.1, final_percentage_weight=1):
    return np.mean(result_list) * (1 - final_percentage_weight) + mean_final_percent(result_list,
                                                                                     percentage) * final_percentage_weight


def plot_rewards(rewards, name=None, xlabel="Step"):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel(xlabel)
    plt.ylabel('Return of current Episode')

    idxs = calculate_reduced_idxs(len(rewards), 1000)
    rewards = reducePoints(rewards, 1000)

    plt.plot(idxs, rewards)
    # Apply mean-smoothing and plot result
    window_size = len(rewards) // 10
    window_size += 1 if window_size % 2 == 0 else 0
    means = meanSmoothing(rewards, window_size)
    max_val = np.max(means)
    min_val = np.min(means)
    # plt.ylim(min_val, max_val * 1.1)
    plt.plot(idxs, means)
    if name is None:
        plt.savefig("current_test.pdf")
    else:
        plt.savefig(name + "_current.pdf")
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

