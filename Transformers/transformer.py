import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math

class FeedForward(nn.Module):

    def __init__(self, input_dim, hidden_dim=2048, dropout=0.1):
        super().__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.W2(self.dropout(F.relu(self.W1(x))))

class SelfAttention(nn.Module):

    def __init__(self, input_dim, query_dim, value_dim):
        super(SelfAttention, self).__init__()
        self.W_q = nn.Linear(input_dim, query_dim)
        self.W_k = nn.Linear(input_dim, query_dim)
        self.W_v = nn.Linear(input_dim, value_dim)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        n_token, q_dim = Q.shape
        score_matrix = (Q @ K.T) / math.sqrt(q_dim)
        mask = torch.triu(torch.ones_like(score_matrix), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        QK = F.softmax(score_matrix + mask, dim=-1)
        head = QK @ V
        return head

class MultiHeadAttention(nn.Module):

    def __init__(self, input_dim, query_dim, value_dim, n_head):
        super().__init__()
        self.W_q = nn.Linear(input_dim, query_dim * n_head)
        self.W_k = nn.Linear(input_dim, query_dim * n_head)
        self.W_v = nn.Linear(input_dim, value_dim * n_head)
        self.n_head = n_head
        self.re_proj = nn.Linear(value_dim * n_head, input_dim)

    def forward(self, x, pad_mask):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        bs, L, _ = Q.shape

        Q = Q.view(bs, L, self.n_head, -1).transpose(1, 2)
        K = K.view(bs, L, self.n_head, -1).transpose(1, 2)
        V = V.view(bs, L, self.n_head, -1).transpose(1, 2)

        bs, n_head, L, q_dim = Q.shape

        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(q_dim)

        mask = torch.triu(torch.ones((bs, 1, L, L)), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        scores = scores.masked_fill(pad_mask[:, None, None, :].bool(), value=float("-inf"))

        multi_heads = torch.matmul(F.softmax(scores, dim=-1), V)
        multi_heads = multi_heads.transpose(1, 2).reshape(bs, L, -1)
        res = self.re_proj(multi_heads)
        return res

class TransformerBlock(nn.Module):

    def __init__(self, input_dim, query_dim, value_dim, n_head, ffn_hidden):
        super().__init__()
        self.ln1 = nn.LayerNorm(input_dim)
        self.mha = MultiHeadAttention(input_dim, query_dim, value_dim, n_head)
        self.ln2 = nn.LayerNorm(input_dim)
        self.ffn = FeedForward(input_dim, ffn_hidden)

    def forward(self, x, mask):
        p1 = self.ln1(x)
        p1 = self.mha(p1, mask)
        p1 = p1 + x
        p2 = self.ln2(p1)
        p2 = self.ffn(p2)
        return p1 + p2

if __name__ == "__main__":
    L, D = 100, 128
    D_q = D_v = 64
    n_head= 8
    batch_size = 50
    ffn_hidden = D * 4

    lengths = (torch.rand(batch_size) * L).to(int)
    lengths = torch.clamp(lengths, min=int(L * 0.5))
    batch_x = [torch.rand(l.item(), D) for l in lengths]
    batch_x = pad_sequence(batch_x, batch_first=True)
    mask = torch.ones(batch_x.shape[:-1])
    for i, l in enumerate(lengths):
        mask[i, :l.item()] = 0

    model = TransformerBlock(D, D_q, D_v, n_head, ffn_hidden)
    output = model(batch_x, mask)
    print(output.shape)
