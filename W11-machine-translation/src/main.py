from config import Config
from preprocessing import Prepocessing
from dataset import CustomDataset
from model import Model
from trainer import Trainer
from loss import Loss
from tester import Tester

def run_training(cfg):
    pre = Prepocessing(cfg)
    train_set = CustomDataset(cfg, pre.train_set)
    dev_set = CustomDataset(cfg, pre.dev_set)
    test_set = CustomDataset(cfg, pre.test_set)
    model = Model(cfg)
    loss_fn = Loss(cfg)
    tester = Tester(cfg, dev_set, test_set)
    trainer = Trainer(cfg, model, tester=tester, train_set=train_set, loss_fn=loss_fn)
    trainer = trainer.train(cfg.num_epochs, cfg.batch_size)


if __name__ == "__main__":
    cfg = Config("config/config.yaml")
    run_training(cfg)
