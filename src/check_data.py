from config import Config
from preprocessing import Prepocessing
from dataset import CustomDataset
from tokenizer import Tokenizer

from torch.utils.data import DataLoader

def check_pe_max_len(dataloader, check_trg=False):
    src_max = -1
    trg_max = -1
    for sample in dataloader:
        src, trg = sample
        src_max = max(src.shape[-1], src_max)
        if check_trg: trg_max = max(trg.shape[-1], trg_max)
    print(src_max)
    if check_trg: print(trg_max)

def run_training(cfg):
    tokenizer = Tokenizer(cfg)
    pre = Prepocessing(cfg, tokenizer)
    train_set = CustomDataset(cfg, pre.train_set, tokenizer)
    dev_set = CustomDataset(cfg, pre.dev_set, tokenizer)
    test_set = CustomDataset(cfg, pre.test_set, tokenizer)

    train_collate_fn = lambda batch: train_set.collate_fn(batch, for_training=True)
    train_dataloader = DataLoader(train_set,
                                 collate_fn=train_collate_fn,
                                 batch_size=1,
                                 shuffle=True,
                                 drop_last=True)

    check_pe_max_len(train_dataloader)

    eval_set = dev_set
    eval_collate_fn = lambda batch: eval_set.collate_fn(batch, for_training=False)

    eval_dataloader = DataLoader(
        eval_set,
        collate_fn=eval_collate_fn,
        batch_size=1,
        shuffle=False,
        drop_last=False
    )

    dev_dataloader = eval_dataloader
    check_pe_max_len(dev_dataloader, check_trg=False)

    eval_set = test_set
    eval_collate_fn = lambda batch: eval_set.collate_fn(batch, for_training=False)

    eval_dataloader = DataLoader(
        eval_set,
        collate_fn=eval_collate_fn,
        batch_size=1,
        shuffle=False,
        drop_last=False
    )

    test_dataloader = eval_dataloader
    check_pe_max_len(test_dataloader, check_trg=False)


if __name__ == "__main__":
    cfg = Config("config/config.yaml")
    run_training(cfg)
