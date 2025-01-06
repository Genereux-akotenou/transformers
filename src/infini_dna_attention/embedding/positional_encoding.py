import torch
from torch import nn

class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding
    args:
        d_mdel  : dimension of model
        max_len : max sequence length
        device  : the desired device of returned tensor
    """
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1) # math operations require floating-point precision.
                                           # convert to [max_len, 1]

        # PE(pos, 2i) ; PE(pos, 2i+1) 
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]