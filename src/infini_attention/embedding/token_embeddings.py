from torch import nn

class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn.Embedding
    they will dense representation of word using weighted matrix
    """
    def __init__(self, vocab_size, d_model, device):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1, device=device)
