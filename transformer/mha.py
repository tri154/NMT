import torch.nn as nn
import torch
import math

class MultiHeadAttention(nn.Module):

    def __init__(self, cfg):
        super(MultiHeadAttention, self).__init__()
        assert cfg.d_model % cfg.n_heads == 0

        self.device = cfg.device
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model

        self.wq = nn.Linear(self.d_model, self.d_model, bias=False)
        self.wk = nn.Linear(self.d_model, self.d_model, bias=False)
        self.wv = nn.Linear(self.d_model, self.d_model, bias=False)

        self.scale = math.sqrt(self.d_model // self.n_heads)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout != 0 else None

        self.wo = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(self, query, key, value, mask=None):
        Q = self.wq(query)
        K = self.wk(key)
        V = self.wv(value)

        Q = Q.view(*Q.shape[:-1], self.n_heads, -1).permute(0, 2, 1, 3)
        K = K.view(*K.shape[:-1], self.n_heads, -1).permute(0, 2, 1, 3)
        V = V.view(*V.shape[:-1], self.n_heads, -1).permute(0, 2, 1, 3)

        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            attention = attention.masked_fill(mask, float("-inf"))

        attention = torch.softmax(attention, dim=-1)

        if self.dropout is not None:
            attention = self.dropout(attention)

        out = torch.matmul(attention, V)

        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(*out.shape[:2], self.d_model)

        out = self.wo(out)
        return out
