import torch
import math
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

class Tester:
    def __init__(self, cfg, dev_set, test_set):
        self.cfg = cfg
        self.dev_set = dev_set
        self.test_set = test_set

    def cal_score(self, ):
        pass

    def test(self, model, tag):
        pass
