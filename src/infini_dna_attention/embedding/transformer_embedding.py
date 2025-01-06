import torch
from torch import nn
from .positional_encoding import PositionalEncoding
from .token_embeddings import TokenEmbedding

class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    """
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device, padding_idx=1, token_type_vocab_size=2):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model, device, padding_idx)
        # -begin
        # to mact bert based model architecture
        # self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.pos_emb = nn.Embedding(max_len, d_model, device=device)
        # -end
        self.token_type_emb = nn.Embedding(token_type_vocab_size, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x, token_type_ids=None):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)

        # Token type embeddings
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(x, dtype=torch.long, device=x.device)
        token_type_emb = self.token_type_emb(token_type_ids)

        embeddings = tok_emb + pos_emb + token_type_emb
        embeddings = self.layer_norm(embeddings)
        return self.drop_out(embeddings)