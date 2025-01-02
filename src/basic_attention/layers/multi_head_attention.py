from torch import nn
from .scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    compute multi head attention
    args: 
        query : a sentence that we focused on (from decoder)
        key   : every sentence to check relationship of query with (the encoder)
        value : every sentence same with key (encoder)
    """
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.d_model   = d_model
        self.n_head    = n_head

        # y = xW^T + b
        self.w_q       = nn.Linear(self.d_model, self.d_model)
        self.w_k       = nn.Linear(self.d_model, self.d_model)
        self.w_v       = nn.Linear(self.d_model, self.d_model)
        self.w_concat  = nn.Linear(self.d_model, self.d_model)

        # Scaled dot product attention 
        self.attention = ScaledDotProductAttention()

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v) # why?

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. scaled dot product
        out, scores = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO: implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head
            :param tensor: [batch_size, length, d_model]
            :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor   = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

        return tensor

    def concat(self, tensor):
        """
        concat result from each heads
            :param tensor: [batch_size, head, length, d_tensor]
            :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
