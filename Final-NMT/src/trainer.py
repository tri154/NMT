from torch.utils.data import DataLoader
from transformers.optimization import get_linear_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
import torch

class Trainer:
    def __init__(self, cfg, model, tokenizer, tester, train_set, loss_fn):
        self.cfg = cfg
        self.tester = tester
        self.model = model
        self.tokenizer = tokenizer
        self.train_set = train_set
        self.loss_fn = loss_fn

    def prepare_optimizer_scheduler(self, train_dataloader):
        opt = AdamW(self.model.parameters(), lr=self.cfg.lr)
        # update every batch.
        num_updates = len(train_dataloader) * self.cfg.num_epochs
        num_warmups = int(num_updates * self.cfg.num_warmups)
        sched = get_linear_schedule_with_warmup(opt, num_warmups, num_updates)
        return opt, sched

    def train_one_epoch(self, train_dataloader, current_epoch):
        device = self.cfg.device
        self.model.train()
        self.opt.zero_grad()

        total_loss = 0.0
        for idx_batch, (batch_src, batch_trg) in enumerate(train_dataloader):
            batch_src = batch_src.to(device)
            batch_trg = batch_trg.to(device)
            self.opt.zero_grad()
            batch_logits, trg_lengths = self.model(batch_src, batch_trg, is_training=True)
            batch_loss = self.loss_fn.compute_loss(batch_logits, batch_trg, trg_lengths)

            # DEBUG
            print(batch_loss)
            input("break")
            # DEBUG

            batch_loss.backward()

            is_updated = True
            is_evaluated = idx_batch == len(train_dataloader) - 1

            if is_updated:
                clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.opt.step()
                self.opt.zero_grad()
                self.sched.step()

            if is_evaluated:
                d_score = self.tester.test(self.model, self.tokenizer, tag='dev', batch_size=self.cfg.test_batch_size)
                self.cfg.logging(f"batch id: {idx_batch}, Dev result : {d_score}", is_printed=True)
                if d_score.score > self.best_score_dev:
                    self.best_score_dev = d_score.score
                    torch.save(self.model.state_dict(), self.cfg.save_path)

            total_loss += batch_loss.item()

        return total_loss

    def train(self, num_epoches, batch_size):
        train_collate_fn = lambda batch: self.train_set.collate_fn(batch, for_training=True)
        train_dataloader = DataLoader(self.train_set,
                                     collate_fn=train_collate_fn,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     drop_last=True)

        self.opt, self.sched = self.prepare_optimizer_scheduler(train_dataloader)

        self.best_score_dev = 0
        for idx_epoch in range(num_epoches):
            self.cfg.logging(f'epoch {idx_epoch + 1}/{num_epoches} ' + '=' * 100, is_printed=True)

            epoch_loss = self.train_one_epoch(train_dataloader, idx_epoch)

            self.cfg.logging(f"epoch: {idx_epoch + 1}, loss={epoch_loss} .", is_printed=True)

        self.model.load_state_dict(torch.load(self.cfg.save_path, map_location=self.cfg.device))
        t_score  = self.tester.test(self.model, self.tokenizer, tag='test', batch_size=self.cfg.test_batch_size)
        self.cfg.logging(f"Test result: {t_score}", is_printed=True)
