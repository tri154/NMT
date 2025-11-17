from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, cfg, data, tokenizer):
        self.source, self.value = data['source'], data['target']

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        return self.source[idx], self.value[idx]

    def collate_fn(self, batch):
        src, target = list(), list()
        for sample in batch:
            s, t = sample
            src.append(s)
            target.append(t)
