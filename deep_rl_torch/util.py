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

from collections import namedtuple
import torch.nn as nn
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

def apply_to_state_list(func, state_list):
    """Apply function to whole list of states (e.g. concatenate, stack etc.)
    Recursively apply this function to nested dicts."""
    if isinstance(state_list[0], dict):
        return {
                key: apply_to_state_list(func, [state[key] for state in state_list])
                for key in state_list[0]
        }
    else:
        return func(state_list)
        
def apply_to_state(func, state):
    if isinstance(state, dict):
        return apply_rec_to_dict(func, state)
    else:
        return func(state)

def apply_rec_to_dict(func, tensor_dict):
    zipped = zip(tensor_dict.keys(), tensor_dict.values())
    return {
            key: apply_rec_to_dict(func, content) if isinstance(content, dict)
            else func(content) 
            for key, content in zipped
    }
    
    
    #return_dict = {}
    #for key in tensor_dict:
    #    content = tensor_dict[key]
    #    if isinstance(content, dict):
    #        return_dict[key] = apply_rec_to_dict(func, content)
    #    else:
    #        return_dict[key] = func(content)
    #return return_dict

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
        '''Get sizes of all params in `model`'''
        mods = list(self.model.modules())
        sizes = []

        for i in range(1, len(mods)):
            m = mods[i]
            p = list(m.params())
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
        '''Calculate total number of bits to store `model` params'''
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

