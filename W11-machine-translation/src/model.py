import torch
from torch import nn
from models.encoder import Encoder
from models.decoder import Decoder

class Model(nn.Module):

    def __init__(self, cfg, tokenizer):
        super(Model, self).__init__()
        self.cfg = cfg
        self.pad_id = tokenizer.pad_id
        self.encoder = Encoder(cfg, tokenizer)
        self.decoder = Decoder(cfg, tokenizer)

    def forward(self, batch_src, batch_trg):
        # for training only.
        bs, src_max_len = batch_src.shape

        src_lengths = (batch_src != self.pad_id).to(int).sum(dim=1)
        trg_lengths = (batch_trg != self.pad_id).to(int).sum(dim=1)

        src_mask = torch.arange(src_max_len).expand((bs, src_max_len)) >= src_lengths.unsqueeze(1)
        src_mask = torch.masked_fill(src_mask.float(), src_mask, float("-inf"))

        enc_out, enc_hidden= self.encoder(batch_src, src_lengths)

        bs, trg_mlen = batch_trg.shape

        decoder_embs = list()
        for i in range(0, trg_mlen):
            if i == 0:
                hidden = enc_hidden[-1].unsqueeze(0) # ensure last layer
                decoder_input = batch_trg[:, 0].unsqueeze(-1)

            out, hidden = self.decoder(decoder_input, hidden, enc_out, src_mask)
            decoder_embs.append(out)

            if i != trg_mlen - 1:
                # teacher forcing
                decoder_input = batch_trg[:, i].unsqueeze(-1)

        decoder_embs = torch.cat(decoder_embs, dim=1)
        return None
