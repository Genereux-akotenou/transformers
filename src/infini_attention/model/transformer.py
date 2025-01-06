import torch
from torch import nn
from ..model.decoder import Decoder
from ..model.encoder import Encoder

class InfiniteEncoderDecoderTransformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device, batch_size, segment_size):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device,
                               batch_size=batch_size, 
                               segment_size=segment_size)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device,
                               batch_size=batch_size, 
                               segment_size=segment_size)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
    
    def load_from_checkpoint(self, checkpoint_path):
        try:
            self.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self.to(self.device)
            print(f"Model weights loaded from {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading checkpoint: {e}")

    def predict(self, src, tokenizer, max_len=50):
        self.eval()
        src = src.to(self.device)
        src_mask = self.make_src_mask(src)
        
        enc_src = self.encoder(src, src_mask)
        trg = torch.tensor([[self.trg_sos_idx]], dtype=torch.long).to(self.device)
        for _ in range(max_len):
            trg_mask = self.make_trg_mask(trg)
            output = self.decoder(trg, enc_src, trg_mask, src_mask)
            next_token = output.argmax(dim=-1)[:, -1].item()

            trg = torch.cat([trg, torch.tensor([[next_token]], dtype=torch.long).to(self.device)], dim=1)

            if next_token == self.trg_pad_idx:
                break

        decoded_prediction = tokenizer.decode(trg[0].cpu().numpy(), skip_special_tokens=True)
        return decoded_prediction