import torch.nn as nn
import torch.nn.functional as F

def _find_multiple(a, b):
    return (-(a // -b)) * b

class SwiGLU(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        hidden_size = cfg.d_model
        inter = _find_multiple(round(cfg.expansion * hidden_size * 2/3), 256)
        self.gate_up = nn.Linear(hidden_size, inter * 2, bias=False)
        self.down = nn.Linear(inter, hidden_size, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(self.dropout(F.silu(gate) * up))
