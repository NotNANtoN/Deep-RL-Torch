import torch

class Normalizer():
    def __init__(self, input_shape, device, verbose=True):
        self.n = 0
        if verbose:
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
        assert inputs.dtype == torch.float16 or inputs.dtype == torch.float32 or inputs.dtype == torch.float64
        return (inputs - self.mean) / obs_std

    def denormalize(self, inputs):
        obs_std = torch.sqrt(self.var)
        return (inputs.float() * obs_std) + self.mean

    def to(self, device):
        self.mean.to(device)
        self.mean_diff.to(device)
        self.var.to(device)
