from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from torch.optim.lr_scheduler import LambdaLR


class Trainer:
    def __init__(self, cfg, model, tokenizer, tester, train_set, loss_fn):
        self.cfg = cfg
        self.tester = tester
        self.model = model
        self.tokenizer = tokenizer
        self.train_set = train_set
        self.loss_fn = loss_fn

    def prepare_optimizer_scheduler(self, train_dataloader):
        opt = AdamW(self.model.parameters(),
                   lr=self.cfg.lr,
                   betas=(self.cfg.opt_b1, self.cfg.opt_b2),
                   eps=self.cfg.opt_eps
        )
        # update every batch.
        num_steps = len(train_dataloader) * self.cfg.num_epochs
        if self.cfg.num_warmups < 1.0:
            num_warmups = int(num_steps * self.cfg.num_warmups)
        else:
            num_warmups = int(self.cfg.num_warmups)
        # sched = get_linear_schedule_with_warmup(opt, num_warmups, num_steps)

        d_model = self.cfg.d_model

        def noam_lambda(step):
            step = max(step, 1)
            return (
                d_model ** (-0.5) * min(step ** (-0.5), step * num_warmups ** (-1.5))
            )

        sched = LambdaLR(opt, noam_lambda)
        return opt, sched

    def train_one_epoch(self, train_dataloader, current_epoch):
        device = self.cfg.device

        total_loss = 0.0
        tracking_loss = 0.0

        for idx_batch, (batch_src, batch_trg) in enumerate(train_dataloader):
            batch_src = batch_src.to(device)
            batch_trg = batch_trg.to(device)

            self.model.train()
            batch_logits = self.model(batch_src, trg_teacher=batch_trg[:, :-1].contiguous())
            batch_loss = self.loss_fn.compute_loss(batch_logits, batch_trg[:, 1:].contiguous())

            # DEBUG
            print(batch_loss)
            input("break")
            # DEBUG
            batch_loss.backward()

            is_updated = True
            is_evaluated = idx_batch == len(train_dataloader) - 1
            is_evaluated = is_evaluated or (self.cfg.eval_freq > 0 and idx_batch % self.cfg.eval_freq == 0 and idx_batch != 0)

            if is_updated:
                self.opt.step()
                self.opt.zero_grad()
                self.sched.step()

            if is_evaluated:
                d_score = self.tester.test(self.model, self.tokenizer, tag='dev', batch_size=self.cfg.test_batch_size)
                self.cfg.logging(f"batch id: {idx_batch}, Dev result : {d_score}", is_printed=True, print_time=True)
                if d_score > self.best_score_dev:
                    self.best_score_dev = d_score
                    torch.save(self.model.state_dict(), self.cfg.save_path)

            if idx_batch % self.cfg.print_freq == 0 and idx_batch != 0:
                self.cfg.logging(f"batch id: {idx_batch}, batch loss: {tracking_loss/ self.cfg.print_freq}", is_printed=True, print_time=True)
                tracking_loss = 0.0

            batch_loss_item = batch_loss.item()
            tracking_loss += batch_loss_item
            total_loss += batch_loss_item

        return total_loss / len(train_dataloader)

    def train(self, num_epoches, batch_size):
        train_collate_fn = lambda batch: self.train_set.collate_fn(batch, for_training=True)
        train_dataloader = DataLoader(self.train_set,
                                     collate_fn=train_collate_fn,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     drop_last=True)

        self.opt, self.sched = self.prepare_optimizer_scheduler(train_dataloader)

        self.best_score_dev = -1
        for idx_epoch in range(num_epoches):
            self.cfg.logging(f'epoch {idx_epoch + 1}/{num_epoches} ' + '=' * 100, is_printed=True)

            epoch_loss = self.train_one_epoch(train_dataloader, idx_epoch)

            self.cfg.logging(f"epoch: {idx_epoch + 1}, loss={epoch_loss} .", is_printed=True)

        self.model.load_state_dict(torch.load(self.cfg.save_path, map_location=self.cfg.device))
        t_score  = self.tester.test(self.model, self.tokenizer, tag='test', batch_size=self.cfg.test_batch_size)
        self.cfg.logging(f"Test result: {t_score}", is_printed=True)
