import torch
import math
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu
# from sacrebleu.metrics import BLEU
import sacrebleu

class Tester:
    def __init__(self, cfg, dev_set, test_set):
        self.cfg = cfg
        self.dev_set = dev_set
        self.test_set = test_set

    def cal_score(self, preds, labels):
        return sacrebleu.corpus_bleu(preds, [labels])

    def test(self, model, tokenizer, tag, batch_size):
        assert tag in ["dev", "test"]
        model.eval()
        device = self.cfg.device
        eval_set = self.dev_set if tag == "dev" else self.test_set
        eval_collate_fn = lambda batch: eval_set.collate_fn(batch, for_training=False)
        eval_dataloader = DataLoader(eval_set,
                                     collate_fn=eval_collate_fn,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     drop_last=False)

        preds = []
        labels = []
        for idx_batch, (batch_src, batch_trg) in enumerate(eval_dataloader):
            batch_src = batch_src.to(device)
            with torch.no_grad():
                batch_preds = model(batch_src, None, is_training=False)
                preds.extend(batch_preds)
                labels.extend(batch_trg)
        preds = tokenizer.detokenize(preds)
        score = self.cal_score(preds, labels)
        return score
