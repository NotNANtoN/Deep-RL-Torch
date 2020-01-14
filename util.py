import logging
import matplotlib
import numpy as np
import os
import random
import tracemalloc
import linecache
import sys
import torch

matplotlib.use('Pdf')
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
from collections import namedtuple, defaultdict
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

def apply_rec_to_dict(func, tensor_dict):
    return_dict = {}
    for key in tensor_dict:
        content = tensor_dict[key]
        if isinstance(content, dict):
            return_dict[key] = apply_rec_to_dict(func, content)
        else:
            return_dict[key] = func(content)
            #tensor_dict[key] = func(content)
    return return_dict

class Normalizer():
    def __init__(self, input_shape, device):
        self.n = 0
        print("Normalizer shape: ", input_shape)
        self.mean = torch.zeros(input_shape, device=device)
        self.mean_diff = torch.zeros(input_shape, device=device)
        self.var = torch.ones(input_shape, device=device)


    def observe(self, x):
        x = x.float().detach()
        self.n += 1.
        last_mean = self.mean.clone()
        assert self.mean.shape == x.shape[1:], "Mean and Input have different shapes. Mean shape: " + str(self.mean.shape) + " X shape: " + str(x.shape)
        self.mean += (x - self.mean).mean(dim=0) / self.n
        self.mean_diff += (x - last_mean).mean(dim=0) * (x - self.mean).mean(dim=0)
        self.var = torch.clamp(self.mean_diff / self.n, min=1e-2)

    def normalize(self, inputs):
        obs_std = torch.sqrt(self.var)
        return (inputs.float() - self.mean) / obs_std

    def denormalize(self, inputs):
        obs_std = torch.sqrt(self.var)
        return (inputs.float() * obs_std) + self.mean

    def to(self, device):
        self.mean.to(device)
        self.mean_diff.to(device)
        self.var.to(device)


class Log(object):
    def __init__(self, path, log, comment, env_name=""):
        self.do_logging = log

        self.comment = comment
        if log:
            self.writer = SummaryWriter(comment=comment)#, log_dir="runs/" + env_name)
        self.episodic_storage = {}
        self.storage = {}
        self.short_term_storage = defaultdict(int)
        self.short_term_count = defaultdict(int)
        self.global_step = 0
        self.tb_path = 'runs'
        #self.run_tb()

    def flush_episodic(self):
        self.episodic_storage = {}

    def get_episodic(self, name):
        return self.episodic_storage[name]

    def _add_to_storage(self, storage, name, value):
        try:
            storage[name].append(value)
        except KeyError:
            storage[name] = [value]

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

    def store_running_mean(self, storage, name, value, count):
        n = count[name]

        if name in storage:
        #        try:
            storage[name] = (storage[name] * (n - 1) + value) / n
        else:
            storage[name] = value
        #except KeyError:
        #    storage[name] = value
        #except TypeError:
        #    storage[name] = value
        return storage

    def add(self, name, value, distribution=False, make_distribution=False, steps=None, skip_steps=0,
            store_episodic=False):
        #self._add_to_storage(self.storage, name, value)
        if store_episodic:
            self._add_to_storage(self.episodic_storage, name, value)

        if self.do_logging:
            self.short_term_count[name] += 1

            if make_distribution:
                if name not in self.short_term_storage:
                    self.short_term_storage[name] = [value]
                else:
                    self.short_term_storage[name].append(value)
            else:
                #print("value: ", value)
                #print(name, "before: ", self.short_term_storage[name])
                self.short_term_storage = self.store_running_mean(self.short_term_storage, name, value, self.short_term_count)
                #print("after: ", self.short_term_storage[name])
            #if "Gradient" in name:
            #    print(name)
            #    print(value)
            #    print(self.short_term_storage[name])
            if self.short_term_count[name] >= skip_steps:
                tb_value = self.short_term_storage[name]#np.mean(self.short_term_storage[name])
                if not isinstance(tb_value, torch.Tensor):
                    tb_value = torch.tensor(tb_value)
                tb_value = tb_value.detach()
                del self.short_term_storage[name]
                del self.short_term_count[name]
                #print(name)
                #print(tb_value)
                if steps is None:
                    steps = self.global_step
                if distribution or make_distribution:
                    add_func = self.writer.add_histogram
                else:
                    add_func = self.writer.add_scalar
                add_func(name, tb_value, global_step=steps)

    def get(self):
        return self.storage

    def step(self):
        self.global_step += 1

    def flush(self):
        if self.do_logging:
            self.writer.flush()

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


class SizeEstimator(object):

    def __init__(self, model, input_size=(1, 1, 32, 32), bits=32):
        '''
        Estimates the size of PyTorch models in memory
        for a given input size
        '''
        self.model = model
        self.input_size = input_size
        self.bits = bits

    def get_parameter_sizes(self):
        '''Get sizes of all parameters in `model`'''
        mods = list(self.model.modules())
        sizes = []

        for i in range(1, len(mods)):
            m = mods[i]
            p = list(m.parameters())
            for j in range(len(p)):
                sizes.append(np.array(p[j].size()))

        self.param_sizes = sizes

    def get_output_sizes(self):
        '''Run sample input through each layer to get output sizes'''
        input_ = torch.rand(self.input_size)
        mods = list(self.model.modules())
        out_sizes = []
        for m in mods:
            out = m(input_)
            out_sizes.append(np.array(out.size()))
            input_ = out

        self.out_sizes = out_sizes

    def calc_param_bits(self):
        '''Calculate total number of bits to store `model` parameters'''
        total_bits = 0
        for i in range(len(self.param_sizes)):
            s = self.param_sizes[i]
            bits = np.prod(np.array(s)) * self.bits
            total_bits += bits
        self.param_bits = total_bits

    def calc_forward_backward_bits(self):
        '''Calculate bits to store forward and backward pass'''
        #total_bits = 0
        #for i in range(len(self.out_sizes)):
        #    s = self.out_sizes[i]
        #    bits = np.prod(np.array(s)) * self.bits
        #    total_bits += bits
        # multiply by 2 for both forward AND backward
        in_ = torch.rand(self.input_size)
        self.model(in_, calc_size=True)


        self.forward_backward_bits =  self.model.total_size * 2

    def calc_input_bits(self):
        '''Calculate bits to store input'''
        self.input_bits = np.prod(np.array(self.input_size)) * self.bits
        return

    def estimate_size(self):
        '''Estimate model size in memory in megabytes and bits'''
        self.get_parameter_sizes()
        #self.get_output_sizes()
        self.calc_param_bits()
        self.calc_forward_backward_bits()
        #self.calc_input_bits()
        total = self.param_bits + self.forward_backward_bits * 32# + self.input_bits

        total_megabytes = (total / 8) / (1024 ** 2)
        return total_megabytes, total




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x, calc_size=False):
        total_size = 0
        if calc_size:
            total_size += np.prod(x.size())
            print(x.size())
        x = self.conv1(x)
        x = F.relu(x)
        if calc_size:
            total_size += np.prod(x.size())
            print(x.size())

        x = self.conv2(x)
        if calc_size:
            total_size += np.prod(x.size())
            print(x.size())

        x = F.max_pool2d(x, 2)
        if calc_size:
            total_size += np.prod(x.size())
            print(x.size())

        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        if calc_size:
            total_size += np.prod(x.size())
            print(x.size())

        x = self.fc1(x)
        x = F.relu(x)
        if calc_size:
            total_size += np.prod(x.size())
            print(x.size())

        x = self.dropout2(x)
        x = self.fc2(x)
        if calc_size:
            total_size += np.prod(x.size())
            print(x.size())

        output = F.log_softmax(x, dim=1)
        self.total_size = total_size
        return output
