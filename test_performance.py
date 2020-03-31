import torch
import numpy as np

class LazyFrames:
    def __init__(self, frames, obs_is_dict, stack_dim):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None
        self.obs_is_dict = obs_is_dict
        self.stack_dim = stack_dim

    def _force(self):
        if self._out is None:
            self._out = self.stack_frames(self._frames)
            self._frames = None
        return self._out

    def stack_frames(self, frames):
        if self.obs_is_dict:
            obs = {
                self.stack([frame[key] for frame in frames])
                for key in frames[0]
            }
        else:
            obs = self.stack(frames)
        return obs

    def stack(self, frames):
        return torch.cat(list(frames), dim=self.stack_dim)

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.type(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class Test:
    def __init__(self):
        self.out = None
        self.out_tensor = None
        self.dtype = torch.float

    def __array__(self):
        if self.out is None:
            self.out = np.array([1,2,3])
        return self.out

    def __tensor__(self):
        if self.out_tensor is None:
            self.out_tensor = torch.tensor([1,2,3])
        return self.out_tensor

    def __dtype__(self):
        return torch.float




a = Test()
print("a: ", a)
b = np.array(a)
print("b: ", b)
c = torch.tensor(a)
print("c: ", c)