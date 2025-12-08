import torch.nn as nn
import torch
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, cfg, tokenizer):
        super(Encoder, self).__init__()
        self.cfg = cfg

        vocab_size = len(tokenizer.src_vocab)

        self.embedding = nn.Embedding(vocab_size, self.cfg.emb_dim)

    def forward(self, batch_input, lengths):
        embs = self.embedding(batch_input)
        pass
