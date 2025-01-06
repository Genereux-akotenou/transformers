from torch import nn
from ..blocks.encoder_layer import EncoderLayer
from ..embedding.transformer_embedding import TransformerEmbedding

class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device, batch_size, segment_size, padding_idx):
        super().__init__()
        self.embedding = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device,
                                        padding_idx=padding_idx)

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head, drop_prob=drop_prob, batch_size=batch_size, segment_size=segment_size) for _ in range(n_layers)]
        )

    def forward(self, x, src_mask=None):
        x = self.embedding(x)
        
        for layer in self.layers:
            x = layer(x, src_mask)

        return x