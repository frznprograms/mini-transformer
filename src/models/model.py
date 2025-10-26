import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.helpers import causal_mask


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, drop: float = 0.1):
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
        self.dropout = nn.Dropout(p=drop)

        self.register_buffer("mask_cache", None, persistent=False)

    def forward(self, x) -> torch.Tensor:
        B, T, D = x.size()
        q = self.WQ(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.WK(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.WV(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.d_k**0.5)

        if self.mask_cache is None or self.mask_cache.size(1) != T:
            self.mask_cache = (
                causal_mask(T).unsqueeze(0).unsqueeze(0).to(x.device)
            )  # (1, 1, T, T)
        mask = self.mask_cache

        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        output = attn @ v
        output = output.transpose(1, 2).contiguous().view(B, T, D)

        return self.Wo(output)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, drop: float = 0.1):
        super().__init__()
        self.attn = SelfAttention(d_model=d_model, n_heads=n_heads, drop=drop)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x) -> torch.Tensor:
        # attention, residual connection, layernorm
        attn_output = self.attn(x)
        x = self.norm1(x + attn_output)

        # feedforward network, residual connection, layernorm
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)

        return x


class MiniTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 27,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 2,
        max_len: int = 128,
        drop: float = 0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, drop=drop)
                for i in range(n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.out_layer = nn.Linear(d_model, vocab_size)

        self.apply(self._init_weights)

    def forward(self, x) -> torch.Tensor:
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # to match batch size
        x = self.token_emb(x) + self.pos_emb(positions)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.out_layer(x)

        return logits

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
