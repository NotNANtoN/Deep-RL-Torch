from collections import defaultdict

import torch
from torch.utils.tensorboard import SummaryWriter


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
        
    def __getitem__(self, key):
        key = self.transform_name(key)
        return self.storage[key]
    
    def __iter__(self):
        for key in self.storage:
            yield key

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
        if not distribution and not make_distribution:
            self._add_to_storage(self.storage, name, value)
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
