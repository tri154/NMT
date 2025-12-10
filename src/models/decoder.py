import torch.nn as nn
import torch
import torch.nn.functional as F

from models.mha import MultiHeadAttention
from models.ffn import FeedForwardLayer
from models.pe import PositionalEncoding

class DecoderLayer(nn.Module):

    def __init__(self, cfg):
        super(DecoderLayer, self).__init__()
        self.ln = nn.LayerNorm(cfg.d_model)
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)

        self.masked_mha = MultiHeadAttention(cfg)
        self.mha = MultiHeadAttention(cfg)

        self.ffn = FeedForwardLayer(cfg)

        self.dropout = nn.Dropout(cfg.dropout)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)

    def forward(self, batch_input, enc_out, src_mask, trg_mask):
        x = self.ln(
            batch_input + self.dropout(self.masked_mha(batch_input, batch_input, batch_input, trg_mask))
        )

        x = self.ln1(
            x + self.dropout1(self.mha(x, enc_out, enc_out, src_mask))
        )

        x = self.ln2(
            x + self.dropout2(self.ffn(x))
        )

        return x


class Decoder(nn.Module):

    def __init__(self, cfg, tokenizer):
        super(Decoder, self).__init__()
        self.cfg = cfg
        vocab_size = len(tokenizer.trg_vocab)

        self.embedding = nn.Embedding(vocab_size, cfg.d_model)
        self.pe = PositionalEncoding(cfg)

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(cfg) for _ in range(cfg.n_decoder_layers)]
        )

    def forward(self, batch_input, enc_out, src_mask, trg_mask):
        x = self.embedding(batch_input)
        x = self.pe(x)

        for layer in self.decoder_layers:
            x = layer(x, enc_out, src_mask, trg_mask)
        return x
