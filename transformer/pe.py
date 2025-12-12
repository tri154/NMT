import torch.nn as nn
import torch
import math

class PositionalEncoding(nn.Module):

    def __init__(self, cfg):
        super(PositionalEncoding, self).__init__()
        self.max_seq_len = cfg.pe_max_seq_len
        self.d_model = cfg.d_model
        self.cfg = cfg

        pe = torch.zeros(self.max_seq_len, self.d_model)

        for pos in range(self.max_seq_len):
            for i in range(0, self.d_model, 2):
                pe[pos, i] = math.sin(pos/(10000**(2*i/self.d_model)))
                pe[pos, i+1] = math.cos(pos/(10000**((2*i+1)/self.d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        @torch.jit.script
        def slice(source, target):
            length = target.size(1);
            return source[:, :length]

        self.slice = slice
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, batch_input):
        input_len = batch_input.shape[1]
        if input_len > self.max_seq_len:
            self.cfg.logging("Input length is longer than max length supported by positional encoding.")
            batch_input = batch_input[:, :self.max_seq_len]

        x = batch_input * math.sqrt(self.d_model)

        pe = self.slice(self.pe, x)
        pe.requires_grad_(False)

        x = x + pe
        x = self.dropout(x)
        return x
