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

    def forward(self, batch_input, mask):
        x = self.dropout(self.mha(batch_input, batch_input, batch_input, mask))
        x = x + batch_input
        x = self.mha_ln(x)

        x1 = self.dropout1(self.ffn(x))
        x1 = x1 + x
        x1 = self.ffn_ln(x1)

        return x1


class Encoder(nn.Module):

    def __init__(self, cfg, tokenizer):
        super(Encoder, self).__init__()
        self.cfg = cfg
        self.n_layers = cfg.n_encoder_layers
        vocab_size = len(tokenizer.src_vocab)

        self.embedding = nn.Embedding(vocab_size, cfg.d_model)
        self.pe = PositionalEncoding(cfg)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(cfg) for _ in range(self.n_layers)]
        )

    def forward(self, batch_input, mask):
        embs = self.embedding(batch_input)
        embs = self.pe(embs)

        for layer in self.encoder_layers:
            embs = layer(embs, mask)

        return embs
