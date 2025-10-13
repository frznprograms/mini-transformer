import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert (
            d_model % n_heads == 0
        ), "Model dimension must be divisible by number of attention heads!"
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None) -> torch.Tensor:
        B, T, D = x.size()
        q = self.WQ(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.WK(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.WV(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.d_k**0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn_scores, dim=-1)

        output = attn @ v
        output = output.transpose(1, 2).contiguous().view(B, T, D)

        return self.Wo(output)
