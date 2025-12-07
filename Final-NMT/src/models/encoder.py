import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):

    def __init__(self, cfg, tokenizer):
        super(Encoder, self).__init__()
        self.cfg = cfg
        # src vocab
        vocab_size = len(tokenizer.src_vocab)

        self.embedding = nn.Embedding(vocab_size, self.cfg.emb_dim)
        # wt
        # self.gru = nn.GRU(self.cfg.emb_dim, self.cfg.hidden_dim, batch_first=True)
        self.gru = nn.GRU(self.cfg.emb_dim, self.cfg.emb_dim, batch_first=True)

    def forward(self, batch_input, lengths):
        embs = self.embedding(batch_input)
        embs = pack_padded_sequence(embs, lengths, batch_first=True, enforce_sorted=False)
        out, h_n = self.gru(embs)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return out, h_n
