import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cross_entropy = nn.CrossEntropyLoss()

    def compute_loss(self, logits, labels, lengths):
        # tackle padding.
        bs, mlen = labels.shape
        mask = torch.arange(mlen).expand((bs, mlen)) < lengths.unsqueeze(-1)
        logits = logits[mask]
        labels = labels[mask]
        loss = self.cross_entropy(logits, labels)
        return loss
