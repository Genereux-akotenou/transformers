from torch import nn
from .positional_encoding import PositionalEncoding
from .token_embeddings import TokenEmbedding

class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    """
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model, device)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)
