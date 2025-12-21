import torch.nn as nn

from transformer import MultiHeadAttention
# from transformer import FeedForwardLayer
from transformer import SwiGLU
from transformer import PositionalEncoding

class DecoderLayer(nn.Module):

    def __init__(self, cfg):
        super(DecoderLayer, self).__init__()
        self.ln = nn.RMSNorm(cfg.d_model)
        self.ln1 = nn.RMSNorm(cfg.d_model)
        self.ln2 = nn.RMSNorm(cfg.d_model)

        self.masked_mha = MultiHeadAttention(cfg)
        self.mha = MultiHeadAttention(cfg)

        self.ffn = SwiGLU(cfg)

        self.dropout = nn.Dropout(cfg.dropout)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)

        self.use_pre_norm = cfg.pre_norm
        self.fn = self.pre_norm if self.use_pre_norm else self.post_norm


    def post_norm(self, batch_input, enc_out, src_mask, trg_mask):
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

    def pre_norm(self, batch_input, enc_out, src_mask, trg_mask):
        x_norm = self.ln(batch_input)
        x = batch_input + self.dropout(
            self.masked_mha(
                x_norm,
                x_norm,
                x_norm,
                trg_mask
            )
        )

        x = x + self.dropout1(
            self.mha(
                self.ln1(x),
                enc_out,
                enc_out,
                src_mask
            )
        )

        x = x + self.dropout2(
            self.ffn(self.ln2(x))
        )

        return x


    def forward(self, batch_input, enc_out, src_mask, trg_mask):
        return self.fn(batch_input, enc_out, src_mask, trg_mask)


class Decoder(nn.Module):

    def __init__(self, cfg, tokenizer):
        super(Decoder, self).__init__()
        self.cfg = cfg
        # vocab_size = len(tokenizer.vocab)

        # self.embedding = nn.Embedding(vocab_size, cfg.d_model)
        self.pe = PositionalEncoding(cfg)

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(cfg) for _ in range(cfg.n_decoder_layers)]
        )
        self.norm = None
        if cfg.pre_norm:
            self.norm = nn.RMSNorm(cfg.d_model)

    def forward(self, batch_embs, enc_out, src_mask, trg_mask):
        # x = self.embedding(batch_input)
        x = self.pe(batch_embs)

        for layer in self.decoder_layers:
            x = layer(x, enc_out, src_mask, trg_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x
