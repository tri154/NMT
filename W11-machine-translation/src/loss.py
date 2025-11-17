import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss:
    def __init__(self, cfg):
        self.cfg = cfg

    def loss(self, pred, label):
        pass
