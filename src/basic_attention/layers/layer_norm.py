import torch
from torch import nn

class LayerNorm(nn.Module):
    """
    A custom implementation of Layer Normalization (LayerNorm)
    """
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta  = nn.Parameter(torch.zeros(d_model))
        self.eps   = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var  = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps) # to avoid 'division by zero' error
        out = self.gamma * out + self.beta
        return out
