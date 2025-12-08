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

    def forward(self, batch_src, batch_trg, is_training):
        src_lengths = (batch_src != self.pad_id).to(int).sum(dim=1)
        enc_out = self.encoder(batch_src, src_lengths)
        breakpoint()
