import torch
from torch import nn

class Model(nn.Module):

    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        self.linear = nn.Linear(1, 2)

    def forward(self, batch_input):
        pred = torch.zeros_like(batch_input)

        return pred
