from collections import defaultdict
import os

import torch
from torch.utils.tensorboard import SummaryWriter


class Log(object):
    def __init__(self, path, log, comment, env_name=""):
        self.do_logging = log

        self.comment = comment
        if log:
            wd = os.getcwd()
            print("WORKING DIRECTORY: ", wd)
            self.writer = SummaryWriter(comment=comment)#, log_dir=os.path.join(wd, "runs/"))#, log_dir="runs/" + env_name)
        self.episodic_storage = {}
        self.storage = {}
        self.short_term_storage = defaultdict(int)
        self.short_term_count = defaultdict(int)
        self.global_step = 0
        self.tb_path = 'runs'
        self.log_steps = None
        
    def __getitem__(self, key):
        key = self.transform_name(key)
        return self.storage[key]
    
    def __iter__(self):
        for key in self.storage:
            yield key

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
            value = value.cpu().detach().item()
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

    def do_save_log(self, name, skip_steps=False, factor=1):
        return not skip_steps or self.short_term_count[name] >= self.log_steps * factor

    def add_count(self, name):
        self.short_term_count[name] += 1

    def is_available(self, name, factor=1):
        self.add_count(name)
        if self.do_save_log(name, skip_steps=True, factor=factor):
            del self.short_term_count[name]
            return True
        else:
            return False

    def add(self, name, value, distribution=False, make_distribution=False, steps=None, skip_steps=False,
            store_episodic=False):
        if not distribution and not make_distribution:
            self._add_to_storage(self.storage, name, value)
        if store_episodic:
            self._add_to_storage(self.episodic_storage, name, value)

        if self.do_logging:
            self.add_count(name)
            # Save in RAM:
            if make_distribution:
                if name not in self.short_term_storage:
                    self.short_term_storage[name] = [value]
                else:
                    self.short_term_storage[name].append(value)
            else:
                self.short_term_storage[name] = self.calc_running_mean(name, value)

            if self.do_save_log(name, skip_steps):
                # Save in Tensorboard:
                tb_value = self.short_term_storage[name]
                if not isinstance(tb_value, torch.Tensor):
                    tb_value = torch.tensor(tb_value)
                tb_value = tb_value.detach()
                del self.short_term_storage[name]
                del self.short_term_count[name]

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
