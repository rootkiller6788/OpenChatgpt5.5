# src/layers/dense_block.py
import torch.nn as nn

from .common import LayerNorm
from .mla_attention import MLAHiddenAttention


class DenseSharedFFN(nn.Module):
    """稠密共享FFN — GPT-5.5 主干核心，占总参 45%"""

    def __init__(self, dim=8192, mlp_ratio=4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class DenseTransformerBlock(nn.Module):
    """稠密 Transformer 块 — Pre-LN + MLA + 残差 + FFN，堆叠 48 层构成主干"""

    def __init__(self, dim=8192, heads=64):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = MLAHiddenAttention(dim, heads)
        self.norm2 = LayerNorm(dim)
        self.ffn = DenseSharedFFN(dim)

    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x
