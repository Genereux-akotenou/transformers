import torch
from torch import nn
from ..blocks.decoder_layer import DecoderLayer
from ..embedding.transformer_embedding import TransformerEmbedding

class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device, batch_size, segment_size, padding_idx):
        super().__init__()
        self.embedding = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device,
                                        padding_idx=padding_idx)

        self.layers = nn.ModuleList(
            [DecoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head, drop_prob=drop_prob, batch_size=batch_size, segment_size=segment_size) for _ in range(n_layers)]
        )

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc, trg_mask, src_mask):
        trg = self.embedding(trg)

        for layer in self.layers:
            trg = layer(trg, enc, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg)
        return output