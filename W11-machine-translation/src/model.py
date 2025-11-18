import torch
from torch import nn
from models.encoder import Encoder
from models.decoder import Decoder

class Model(nn.Module):

    def __init__(self, cfg, tokenizer):
        super(Model, self).__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg, tokenizer)
        self.decoder = Decoder(cfg, tokenizer)

    def forward(self, batch_src, batch_trg):
        enc_out, _ = self.encoder(batch_src)

        return None
