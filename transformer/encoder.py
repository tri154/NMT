import torch.nn as nn
from transformer import PositionalEncoding
from transformer import MultiHeadAttention
from transformer import FeedForwardLayer

class EncoderLayer(nn.Module):

    def __init__(self, cfg):
        super(EncoderLayer, self).__init__()
        self.ffn = FeedForwardLayer(cfg)
        self.ffn_ln = nn.LayerNorm(cfg.d_model)
        self.mha = MultiHeadAttention(cfg)
        self.mha_ln = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.use_pre_norm = cfg.pre_norm
        self.fn = self.pre_norm if self.use_pre_norm else self.post_norm

    def post_norm(self, batch_input, mask):
        x = self.dropout(self.mha(batch_input, batch_input, batch_input, mask))
        x = x + batch_input
        x = self.mha_ln(x)

        x1 = self.dropout1(self.ffn(x))
        x1 = x1 + x
        x1 = self.ffn_ln(x1)
        return x1

    def pre_norm(self, batch_input, mask):
        x_norm = self.mha_ln(batch_input)
        x = self.dropout(
            self.mha(x_norm, x_norm, x_norm, mask)
        )
        x = x + batch_input

        x_norm1 = self.ffn_ln(x)
        x1 = self.dropout1(self.ffn(x_norm1))
        x1 = x1 + x

        return x1

    def forward(self, batch_input, mask):
        return self.fn(batch_input, mask)


class Encoder(nn.Module):

    def __init__(self, cfg, tokenizer):
        super(Encoder, self).__init__()
        self.cfg = cfg
        self.n_layers = cfg.n_encoder_layers
        # vocab_size = len(tokenizer.vocab)

        # self.embedding = nn.Embedding(vocab_size, cfg.d_model)
        self.pe = PositionalEncoding(cfg)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(cfg) for _ in range(self.n_layers)]
        )

    def forward(self, batch_embs, mask):
        # embs = self.embedding(batch_input)
        embs = self.pe(batch_embs)

        for layer in self.encoder_layers:
            embs = layer(embs, mask)

        return embs
