import torch.nn as nn

class FeedForwardLayer(nn.Module):

    def __init__(self, cfg):
        super(FeedForwardLayer, self).__init__()
        self.d_model = cfg.d_model
        self.d_ff = cfg.d_ff
        self.w1 = nn.Linear(self.d_model, self.d_ff, bias=False)
        self.w2 = nn.Linear(self.d_ff, self.d_model, bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, batch_input):
        return self.w2(self.dropout(self.relu(self.w1(batch_input))))
