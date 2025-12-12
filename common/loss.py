import torch.nn as nn

class Loss:
    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    def compute_loss(self, logits, labels):
        # tackle padding.
        bs, mlen, vocab_size = logits.shape
        loss = self.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1)
        )

        return loss
