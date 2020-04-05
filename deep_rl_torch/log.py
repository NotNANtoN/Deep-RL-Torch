from collections import defaultdict
import os

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class Log(object):
    def __init__(self, path, log, comment):
        self.do_logging = log

        self.comment = comment
        if log:
            wd = os.getcwd()
            self.writer = SummaryWriter(comment=comment)#, log_dir=os.path.join(wd, "runs/"))#, log_dir="runs/" + env_name)
        self.episodic_storage = {}
        self.storage = {}  # Stores all infos for later plotting
        self.short_term_storage = defaultdict(int)  # Calculates running mean
        self.short_term_count = defaultdict(int)  # Counts for calculating running mean
        self.distr_count = defaultdict(int)  # Count for make_distribution to check when enough sample are gathered
        self.step_dict = defaultdict(list)  # Saves for each variable when the values where logged
        self.global_step = 0
        self.log_steps = None
        # Keep track of ep lens:
        self.mean_ep_len = 100
        self.total_eps = 1
        self.ep_len_count = defaultdict(int)

    def reset(self):
        if self.do_logging:
            wd = os.getcwd()
            self.writer = SummaryWriter(comment=self.comment)#, log_dir=os.path.join(wd, "runs/"))#, log_dir="runs/" + env_name)
        self.episodic_storage = {}
        self.storage = {}  # Stores all infos for later plotting
        self.short_term_storage = defaultdict(int)  # Calculates running mean
        self.short_term_count = defaultdict(int)  # Counts for calculating running mean
        self.distr_count = defaultdict(int)  # Count for make_distribution to check when enough sample are gathered
        self.step_dict = defaultdict(list)  # Saves for each variable when the values where logged
        self.global_step = 0
        self.log_steps = None
        # Keep track of ep lens:
        self.mean_ep_len = 100
        self.total_eps = 1
        self.ep_len_count = defaultdict(int)
        
    def __getitem__(self, key):
        key = self.transform_name(key)
        return np.array(self.storage[key])
    
    def __setitem__(self, key, val):
        key = self.transform_name(key)
        self.storage[key] = val
    
    def __iter__(self):
        for key in self.storage:
            yield key

    def count_eps_step(self, source):
        self.ep_len_count[source] += 1

    def count_eps(self, source):
        self.total_eps += 1
        self.mean_ep_len = (self.mean_ep_len * (self.total_eps - 1) + self.ep_len_count[source]) / self.total_eps
        del self.ep_len_count[source]

    def set_log_steps(self, val):
        self.log_steps = val
        print("Log steps: ", val)

    def flush_episodic(self):
        self.episodic_storage = {}

    def get_episodic(self, name):
        name = self.transform_name(name)
        return self.episodic_storage[name]

    def transform_name(self, name):
        return name.replace("/", "_")
        
    def _add_to_storage(self, storage, name, value):
        name = self.transform_name(name)
        if isinstance(value, torch.Tensor):
            value = value.cpu().detach().numpy()
        try:
            storage[name].append(value)
        except KeyError:
            storage[name] = [value]

    def calc_running_mean(self, name, value):
        n = self.short_term_count[name]
        storage = self.short_term_storage
        if name in storage:
            value = (storage[name] * (n - 1) + value) / n
        return value

    def do_save_log(self, name, use_skip, skip_steps=0, factor=1, make_distr=False, distr_steps=0):
        if make_distr:
            return self.distr_count[name] >= distr_steps
        else:
            threshold = skip_steps if skip_steps else self.log_steps * factor
            return not use_skip or self.short_term_count[name] >= threshold

    def add_count(self, name):
        self.short_term_count[name] += 1

    def is_available(self, name, factor=1, reset=True, skip_steps=0):
        self.add_count(name)
        if self.do_save_log(name, use_skip=True, skip_steps=skip_steps, factor=factor):
            if reset:
                del self.short_term_count[name]
            return True
        else:
            return False

    def add(self, name, value, distribution=False, make_distr=False, distr_steps=0, steps=None, use_skip=False, skip_steps=0,
            store_episodic=False):
        # Add to episodic storage (flushed after every episode):
        if store_episodic:
            self._add_to_storage(self.episodic_storage, name, value)

        if self.do_logging:
            self.add_count(name)
            # Save in RAM:
            if make_distr:
                self.distr_count[name] += 1
                if name not in self.short_term_storage:
                    self.short_term_storage[name] = [value]
                else:
                    self.short_term_storage[name].append(value)

            else:
                self.short_term_storage[name] = self.calc_running_mean(name, value)

            if self.do_save_log(name, use_skip, skip_steps=skip_steps, make_distr=make_distr,
                                distr_steps=distr_steps):
                # Get value to store long term:
                tb_value = self.short_term_storage[name]
                if not isinstance(tb_value, torch.Tensor):
                    tb_value = torch.tensor(tb_value)
                tb_value = tb_value.detach()
                del self.short_term_storage[name]
                del self.short_term_count[name]
                if make_distr:
                    del self.distr_count[name]
                # Set steps if not specified:
                if steps is None:
                    steps = self.global_step
                # Save to RAM:
                self._add_to_storage(self.storage, name, tb_value)
                # Also save time point of logging
                self._add_to_storage(self.step_dict, name, steps)

                # Save in Tensorboard:
                if distribution or make_distr:
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

    def cleanse(self):
        """This method deletes everything from this object that is not essential for plotting."""
        del self.writer
        del self.episodic_storage
        del self.short_term_storage
        del self.short_term_count
        del self.distr_count
        del self.ep_len_count
