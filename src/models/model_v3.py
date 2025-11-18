import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from src.utils.helpers import causal_mask


class SelfAttentionV3(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, drop: float = 0.1, max_len: int = 128
    ):
        super().__init__()
        assert d_model % n_heads == 0, (
            "Model dimension must be divisible by number of attention heads!"
        )
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=drop)
        self.max_len = max_len

        self.register_buffer("mask_cache", None, persistent=False)
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)

    def forward(self, x) -> torch.Tensor:
        B, T, D = x.size()
        q = self.WQ(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.WK(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.WV(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # build/retrieve rope cache
        if self.cos_cached is None or self.cos_cached.size(2) < T:
            cos, sin = self._build_rope_cache(T, x.device, x.dtype)
            self.cos_cached = cos
            self.sin_cached = sin
        else:
            # truncate to sequence length T just in case
            cos = self.cos_cached[:, :, :T, :]
            sin = self.sin_cached[:, :, :T, :]

        # apply rope
        q = self.apply_rope(q, cos, sin)
        k = self.apply_rope(k, cos, sin)

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

    def _build_rope_cache(self, T: int, device: str, dtype: torch.dtype):
        dim = self.d_k
        half_dim = dim // 2

        # frequencies - applies sinusoidal frequency schedule
        freq_seq = torch.arange(half_dim, device=device, dtype=dtype)
        freq_seq = 1.0 / (10000 ** (2 * freq_seq / dim))  # (, half_dim)

        # positions * frequency = angle of rotation
        pos = torch.arange(T, device=device, dtype=dtype)  # (T,)
        angles = pos.unsqueeze(1) * freq_seq.unsqueeze(0)  # (T, half_dim)

        # precompute sin and cos
        cos = angles.cos()
        sin = angles.sin()

        # reshape for broadcasting to (1, 1, T, half_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        return cos, sin

    def apply_rope(self, x, cos: torch.Tensor, sin: torch.Tensor):
        # x: (B, H, T, D)
        D = x.shape[3]
        half = D // 2

        x1 = x[..., :half]
        x2 = x[..., half:]

        # break down operations efficiently like paper
        rot_x1 = x1 * cos - x2 * sin
        rot_x2 = x1 * sin + x2 * cos

        return torch.cat([rot_x1, rot_x2], dim=-1)


class TransformerBlockV3(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, drop: float = 0.1):
        super().__init__()
        self.attn = SelfAttentionV3(d_model=d_model, n_heads=n_heads, drop=drop)
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


class MiniTransformerV3(nn.Module):
    def __init__(
        self,
        vocab_size: int = 27,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 2,
        drop: float = 0.1,
        max_len: int = 128,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # self.pos_emb = nn.Embedding(max_len, d_model)
        d_k = d_model % n_heads
        if d_k % 2 != 0:
            logger.warning(
                "Per-head dimensionality is not even. RoPE may not perform as expected."
            )

        self.blocks = nn.ModuleList(
            [
                TransformerBlockV3(
                    d_model=d_model, n_heads=n_heads, d_ff=d_ff, drop=drop
                )
                for i in range(n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.out_layer = nn.Linear(d_model, vocab_size)

        self.apply(self._init_weights)

    def forward(self, x) -> torch.Tensor:
        # B, T = x.shape
        # positions = torch.arange(T, device=x.device).unsqueeze(0)  # to match batch size
        # x = self.token_emb(x) + self.pos_emb(positions)

        x = self.token_emb(x)

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
