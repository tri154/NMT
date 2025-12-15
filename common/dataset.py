from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):

    def __init__(self, cfg, data, tokenizer):
        self.source, self.value = data['source'], data['target']
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        return self.source[idx], self.value[idx]

    def collate_fn(self, batch, for_training):
        # training: pad both src and trg, tokenize both src, trg.
        # otherwise: only pad src, only tokenize src.
        pad = self.tokenizer.pad
        src, trg = list(), list()
        mlen_src = max([len(s) for s, t in batch])
        mlen_trg = max([len(t) for s, t in batch])
        for s, t in batch:
            s = s + [pad] * (mlen_src - len(s))
            if for_training:
                t = t + [pad] * (mlen_trg - len(t))
            src.append(s)
            trg.append(t)
        src_idx = self.tokenizer.token2ids(src)
        if for_training:
            trg_idx = self.tokenizer.token2ids(trg)
        else:
            trg_idx = trg

        src_idx = torch.tensor(src_idx)
        if for_training:
            trg_idx = torch.tensor(trg_idx)
        return src_idx, trg_idx
