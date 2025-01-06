import math
from torch import nn

class ScaledDotProductAttention(nn.Module):
    """
    compute scale dot product attention
    args: 
        query : a sentence that we focused on (from decoder)
        key   : every sentence to check relationship of query with (the encoder)
        value : every sentence same with key (encoder)
    """

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        # input (k) is 4 dimension tensor
        batch_size, head, length, d_tensor = k.size()

        # 1. scaled dot product & norm
        k_t = k.transpose(2, 3)
        score = (q @ k_t) / math.sqrt(d_tensor)  

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. softmax 
        score = self.softmax(score)

        # 4. times value
        v = score @ v

        return v, score
