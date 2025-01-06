import torch
from torch import nn
from .scaled_dot_product_attention import ScaledDotProductAttention
from ..utils.compressive_memory import CompressiveMemory
import matplotlib.pyplot as plt

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
        #print(scores.size())

        # 5. visualize attention map
        # TODO: implement visualization
        #self.visualize_attention(scores, head=0)

        return out
    
    def visualize_attention(self, scores, head=0):
        """
        Visualize the attention scores for a single head.
        Args:
            scores: Attention scores of shape (batch_size, n_head, seq_length, seq_length).
            head: Index of the head to visualize.
        """
        # Pick the first batch and the specified head
        attention_map = scores[0, head].detach().cpu().numpy()

        # Plot the attention map
        plt.figure(figsize=(6, 6))
        plt.imshow(attention_map, cmap='viridis')
        plt.colorbar()
        plt.title(f'Attention Map - Head {head}')
        plt.xlabel('Key Sequence')
        plt.ylabel('Query Sequence')
        plt.show()

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

class InfiniAttention(nn.Module):
    """
    compute infinite context attention
    source:
        ["Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention"](https://arxiv.org/abs/2404.07143)
    args:
        d_model: dimension of the input embeddings
        n_head: number of heads used for MHA
        segment_size: size of segments
        use_attn_linear_bias: to use a bias in the linear layer after attention
        use_delta_update_rule: to use the delta update rule mentioned in section 2.1.2(paper)
    """
    def __init__(self, batch_size, d_model, n_head, segment_size, use_attn_linear_bias=None, use_delta_update_rule=None):
        super(InfiniAttention, self).__init__()
        self.batch_size = batch_size
        self.d_model = d_model
        self.n_head = n_head
        self.segment_size = segment_size
        self.use_attn_linear_bias = use_attn_linear_bias
        self.use_delta_update_rule = use_delta_update_rule
        self.segments = None
        self.d_key = self.d_value = self.d_model
        self.memory = CompressiveMemory(batch_size, self.d_key, self.d_value, self.segment_size)
        self.beta = nn.Parameter(torch.randn((1, 1, 1)))
        self.o_proj  = nn.Linear(self.d_model, self.d_model)

    def split(self, tensor):
        batch_size, length, d_tensor = tensor.size()
        segments = torch.split(tensor, self.segment_size, dim=1)
        return segments
    
    def split_mask(self, mask):
        segments = torch.split(mask, self.segment_size, dim=-1)
        return segments

    def forward(self, q, k, v, mask=None):
        # 0. to collect data
        output_segments = []

        # 1. Split the input into segments
        q_segments = self.split(q)
        k_segments = self.split(k)
        v_segments = self.split(v)
        mask_segments = self.split_mask(mask) if mask is not None else None
        #print(len(q_segments))

        for q_seg, k_seg, v_seg, mask_seg in zip(q_segments, k_segments, v_segments, mask_segments):
            # 2: dot-product attention within each segment.
            self.mh_attention = MultiHeadAttention(d_model=self.d_model, n_head=self.n_head)
            A_dot = self.mh_attention(q=q_seg, k=k_seg, v=v_seg, mask=mask_seg)

            # 3: pull Amem from compressive memory using current segment’s query
            A_mem = self.memory.retrieve(q_seg.detach())

            # 4: combine local context with the long-term context
            beta =  nn.Sigmoid()(self.beta)
            A = beta * A_mem + (1 - beta) * A_dot

            # 5: Update compressive memory by adding KV
            self.memory.update(q_seg.detach(), v_seg.detach())

            # 6: discard the previous segment's attention states pass updated memory to next segment
            output = self.o_proj(A)
            output_segments.append(output)

        # concat along sequence dimension
        out = torch.cat(output_segments, dim=1)

        # 7. Free compressive memory
        self.memory.free()

        # 8. visualize attention map
        # TODO: implement visualization

        #print(", ", out.size())
        return out