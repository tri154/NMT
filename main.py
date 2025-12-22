from shared import Config, CustomDataset, Loss, Prepocessing, Tester, Trainer, WordTokenizer, BPETokenizer
from transformer import Model

def run_training(cfg):
    tokenizer = WordTokenizer(cfg) if cfg.tkn_type == "word" else BPETokenizer(cfg)
    pre = Prepocessing(cfg, tokenizer)
    train_set = CustomDataset(cfg, pre.train_set, tokenizer, bidirect=cfg.bidirect)
    cfg.logging(f"train set length: {len(train_set)} .", is_printed=True)
    dev_set = CustomDataset(cfg, pre.dev_set, tokenizer)
    test_set = CustomDataset(cfg, pre.test_set, tokenizer)
    model = Model(cfg, tokenizer).to(cfg.device)
    loss_fn = Loss(cfg, tokenizer)
    tester = Tester(cfg, dev_set, test_set)
    trainer = Trainer(cfg, model, tokenizer, tester=tester, train_set=train_set, loss_fn=loss_fn)
    trainer = trainer.train(cfg.num_epochs, cfg.train_batch_size)

if __name__ == "__main__":
    dft = "configs/config.yaml"
    cfg = Config(dft)
    run_training(cfg)
