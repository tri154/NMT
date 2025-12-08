import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class FeedForwardLayer(nn.Module):

    def __init__(self, cfg):
        super(FeedForwardLayer, self).__init__()
        self.d_model = cfg.d_model
        self.d_ff = cfg.d_ff
        self.w1 = nn.Linear(self.d_model, self.d_ff)
        self.w2 = nn.Linear(self.d_ff, self.d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, batch_input):
        return self.w2(self.dropout(self.relu(self.w1(batch_input))))

class MultiHeadAttention(nn.Module):

    def __init__(self, cfg):
        super(MultiHeadAttention, self).__init__()
        assert cfg.d_model % cfg.n_heads == 0

        self.device = cfg.device
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model

        self.wq = nn.Linear(self.d_model, self.d_model)
        self.wk = nn.Linear(self.d_model, self.d_model)
        self.wv = nn.Linear(self.d_model, self.d_model)

        self.scale = math.sqrt(self.d_model // self.n_heads)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout != 0 else None

        self.wo = nn.Linear(self.d_model, self.d_model)

    def forward(self, batch_input, lengths):
        Q = self.wq(batch_input)
        K = self.wk(batch_input)
        V = self.wv(batch_input)

        bs, mlen, _ = Q.shape

        Q = Q.view(bs, mlen, self.n_heads, -1).permute(0, 2, 1, 3)
        K = K.view(bs, mlen, self.n_heads, -1).permute(0, 2, 1, 3)
        V = V.view(bs, mlen, self.n_heads, -1).permute(0, 2, 1, 3)

        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        padding_mask = torch.arange(mlen, device=self.device).expand(bs, mlen) >= lengths.unsqueeze(-1)
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)

        attention = attention.masked_fill(padding_mask, float("-inf"))

        attention = torch.softmax(attention, dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)

        out = torch.matmul(attention, V)

        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(bs, mlen, self.d_model)

        out = self.wo(out)
        return out

class EncoderLayer(nn.Module):

    def __init__(self, cfg):
        super(EncoderLayer, self).__init__()
        self.ffn = FeedForwardLayer(cfg)
        self.ffn_ln = nn.LayerNorm(cfg.d_model)
        self.mha = MultiHeadAttention(cfg)
        self.mha_ln = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.dropout1 = nn.Dropout(cfg.dropout)

    def forward(self, batch_input, lengths):
        x = self.dropout(self.mha(batch_input, lengths))
        x = x + batch_input
        x = self.mha_ln(x)

        x1 = self.dropout1(self.ffn(x))
        x1 = x1 + x
        x1 = self.ffn_ln(x1)

        return x1


class PositionalEncoding(nn.Module):

    def __init__(self, cfg):
        super(PositionalEncoding, self).__init__()

    def forward(self, batch_input):
        return batch_input

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

    def forward(self, batch_input, lengths):
        embs = self.embedding(batch_input)
        embs = self.pe(embs)

        for layer in self.encoder_layers:
            embs = layer(embs, lengths)
        return embs
