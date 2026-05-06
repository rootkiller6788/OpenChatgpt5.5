# src/layers/dense_block.py
import torch.nn as nn

from .common import LayerNorm
from .mla_attention import LatentCompressedAttention


class DenseFFN(nn.Module):
    """[Speculative] 标准稠密 FFN — Linear → GELU → Linear"""

    def __init__(self, dim=8192, mlp_ratio=4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class DenseTransformerBlock(nn.Module):
    """[Speculative] 标准 Transformer 块 — Pre-LN + Attention + FFN 残差

    层数和维度均为纯推测，无官方证据。
    """

    def __init__(self, dim=8192, heads=64):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = LatentCompressedAttention(dim, heads)
        self.norm2 = LayerNorm(dim)
        self.ffn = DenseFFN(dim)

    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x
