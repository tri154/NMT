import torch
import torch.nn as nn

def rotate_half(tensor):
    dim = tensor.shape[-1]
    x1 = tensor[..., :dim//2]
    x2 = tensor[..., dim//2: ]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary(tensor, cossin):
    cos, sin = cossin
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    applied_tensor = cos * tensor + sin * rotate_half(tensor)
    return applied_tensor


class RotaryEmbedding(nn.Module):
    def __init__(self, hidden_dim, max_position_embeddings, base):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, hidden_dim, 2, dtype=torch.float32) / hidden_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)

        emb = torch.cat([freqs, freqs], dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self, mlen):
        return self.cos_cached[:mlen], self.sin_cached[:mlen]
