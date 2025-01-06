import torch
from torch import nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange

class CompressiveMemory(nn.Module):
    """
    compressive memory system that maintains a fixed-size memory buffer for Transformer layers. 
    This memory stores key-value pairs from previous computations and allows retrieval.
    """
    def __init__(self, batch_size, d_key, d_value, segment_length):
        super(CompressiveMemory, self).__init__()
        self.batch_size = batch_size
        self.d_key = d_key
        self.d_value = d_value
        self.segment_length = segment_length
        self.memory = torch.zeros((batch_size, d_key, d_value))
        self.z_norm = torch.zeros((batch_size, d_key, 1))

    def retrieve(self, query):
        if torch.all(self.memory == 0):
            return torch.zeros((self.batch_size, query.size(1), self.d_value))
        
        sigma_query = F.elu(query) + 1
        print('trace->', sigma_query.size(), self.memory.size())
        retrieved_memory = einsum(
            "bsd,bkv->bsv",
            sigma_query,
            self.memory
        )
        denominator = einsum(
            "bsd,bk->bs",
            sigma_query,
            self.z_norm.squeeze(-1),
        )
        denominator = rearrange(
            denominator,
            "batch_size seq_len -> batch_size seq_len 1",
        )
        retrieved_memory = retrieved_memory / denominator
        return retrieved_memory

    def update(self, query, value):
        sigma_key = F.elu(query) + 1
        sigma_key_t = (F.elu(query) + 1).transpose(-2, -1)

        self.memory += torch.einsum(
            "bkd,bsv->bkv", 
            sigma_key_t,
            value
        )
        self.z_norm += torch.einsum(
            "bsd->bd",
            sigma_key
        ).unsqueeze(-1)

    def dynamically_match_batch_size(self, batch_size):
        # not used
        if self.memory.shape[0] != batch_size:
            self.memory = self.memory.expand(batch_size, -1, -1).clone()
        if self.z.shape[0] != batch_size:
            self.z = self.z.expand(batch_size, -1, -1).clone()

    def free(self):
        self.memory.zero_()
        self.z_norm.zero_()
