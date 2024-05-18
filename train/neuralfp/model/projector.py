import torch.nn as nn
import torch.nn.functional as F


class LinearProjector(nn.Module):
    __constants__ = ['d', 'h']
    def __init__(self, d, h, u):
        super(LinearProjector, self).__init__()
        assert h%d == 0, 'h must be divisible by d'
        self.d = d
        self.h = h
        self.u = u
        self.v = h//d
        self.linear1 = nn.Conv1d(d * self.v, d * u, kernel_size=(1,), groups=d)
        self.elu = nn.ELU()
        self.linear2 = nn.Conv1d(d * u, d, kernel_size=(1,), groups=d)
    
    def forward(self, x, norm=True):
        x = x.reshape([-1, self.h, 1])
        x = self.linear1(x)
        x = self.elu(x)
        x = self.linear2(x)
        x = x.reshape([-1, self.d])
        if norm:
            x = F.normalize(x, p=2.0)
        return x