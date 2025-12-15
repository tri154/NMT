import torch
import torch.nn as nn

class Loss:
    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_id,
            label_smoothing=self.cfg.label_smoothing
        )
        self.vocab_size = len(tokenizer.trg_vocab)
        self.confidence = 1.0 - self.cfg.label_smoothing
        self.pad_id = tokenizer.pad_id

    def compute_loss(self, logits, labels):
        # return self.label_smoothing(logits, labels)
        return self.compute_cross_entropy(logits, labels)

    def compute_cross_entropy(self, logits, labels):
        # tackle padding.
        bs, mlen, vocab_size = logits.shape
        loss = self.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1)
        )

        return loss

    def label_smoothing(self, pred, target):
        bs, mlen, vocab_size = pred.shape
        pred = pred.view(-1, vocab_size)
        target = target.view(-1)
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.cfg.label_smoothing / (self.vocab_size - 2))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.pad_id] = 0
            mask = torch.nonzero(target == self.pad_id)
            if mask.dim() > 0:
                true_dist.index_fill_(0, mask.squeeze(), 0.0)

        return torch.mean(torch.sum(-true_dist * pred, dim=-1))
