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
        position = torch.arange(self.max_seq_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() *
            (-math.log(10000.0) / self.d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

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
            self.cfg.logging("Input length is longer than max length supported by positional encoding.", is_printed=True)
            batch_input = batch_input[:, :self.max_seq_len]

        x = batch_input * math.sqrt(self.d_model)

        pe = self.slice(self.pe, x)
        pe.requires_grad_(False)

        x = x + pe
        x = self.dropout(x)
        return x
