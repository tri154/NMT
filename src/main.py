from config import Config
from preprocessing import Prepocessing
from dataset import CustomDataset
from model import Model
from trainer import Trainer
from loss import Loss
from tester import Tester
from tokenizer import Tokenizer

def run_training(cfg):
    tokenizer = Tokenizer(cfg)
    pre = Prepocessing(cfg, tokenizer)
    train_set = CustomDataset(cfg, pre.train_set, tokenizer)
    dev_set = CustomDataset(cfg, pre.dev_set, tokenizer)
    test_set = CustomDataset(cfg, pre.test_set, tokenizer)
    model = Model(cfg, tokenizer).to(cfg.device)
    loss_fn = Loss(cfg, tokenizer)
    tester = Tester(cfg, dev_set, test_set)
    trainer = Trainer(cfg, model, tokenizer, tester=tester, train_set=train_set, loss_fn=loss_fn)
    trainer = trainer.train(cfg.num_epochs, cfg.train_batch_size)

if __name__ == "__main__":
    cfg = Config("config/config.yaml")
    run_training(cfg)
