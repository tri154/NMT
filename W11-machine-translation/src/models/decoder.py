import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class Attention(nn.Module):

    def __init__(self, cfg):
        super(Attention, self).__init__()

    def forward(self, q, k, v, mask=None):
        simi = q * k
        simi = simi.sum(dim=-1)
        if mask is not None:
            simi = simi + mask
        score = torch.softmax(simi, dim=-1)
        context = score.unsqueeze(-1) * v
        context = context.sum(1)
        return context

class BahdanauAttention(nn.Module):
    def __init__(self, cfg):
        super(BahdanauAttention, self).__init__()
        self.cfg = cfg
        # wt
        # self.W1 = nn.Linear(cfg.hidden_dim, cfg.unit_dim)
        # self.W2 = nn.Linear(cfg.hidden_dim, cfg.unit_dim)
        self.W1 = nn.Linear(cfg.emb_dim, cfg.unit_dim)
        self.W2 = nn.Linear(cfg.emb_dim, cfg.unit_dim)
        self.V = nn.Linear(cfg.unit_dim, 1)

    def forward(self, q, k, v, mask=None):
        simi = torch.tanh(self.W1(q) + self.W2(k))
        simi = self.V(simi).squeeze(-1)
        if mask is not None:
            simi = simi + mask
        score = torch.softmax(simi, dim=-1)
        context = score.unsqueeze(-1) * v
        context = context.sum(1)
        return context

class Decoder(nn.Module):

    def __init__(self, cfg, tokenizer):
        super(Decoder, self).__init__()
        self.cfg = cfg
        # target vocab
        vocab_size = len(tokenizer.trg_vocab)

        self.embedding = nn.Embedding(vocab_size, self.cfg.emb_dim)
        self.attention = BahdanauAttention(cfg)
        # wt
        # self.attention = Attention(cfg)
        # self.gru = nn.GRU(self.cfg.emb_dim + self.cfg.hidden_dim, self.cfg.hidden_dim, batch_first=True)
        self.gru = nn.GRU(self.cfg.emb_dim + self.cfg.emb_dim, self.cfg.emb_dim, batch_first=True)

    def forward(self, batch_input, hidden, enc_out, src_mask=None):
        embs = self.embedding(batch_input)

        hidden = hidden.transpose(0, 1)
        h_0 = self.attention(hidden, enc_out, enc_out, src_mask)
        h_0 = h_0.unsqueeze(1)
        embs = torch.cat([embs, h_0], dim=-1)

        hidden = hidden.transpose(0, 1)
        out, h_n = self.gru(embs, hidden)
        return out, h_n
