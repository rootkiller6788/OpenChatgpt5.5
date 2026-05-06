# src/layers/mla_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentCompressedAttention(nn.Module):
    """[Speculative] 潜空间压缩注意力 — KV 通过线性瓶颈压缩后计算注意力

    Q 使用完整维度，K/V 先线性压缩到潜空间，在压缩空间中计算注意力，
    输出再投影回完整维度。可减少 KV 缓存和计算量。

    当前为 naive 参考实现。没有证据表明 OpenAI 使用这种特定的注意力机制。
    """

    def __init__(self, dim=8192, heads=64, latent_compress=8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.latent_dim = dim // latent_compress
        self.hd = dim // heads
        self.hd_latent = self.latent_dim // heads

        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.k_compress = nn.Linear(dim, self.latent_dim)
        self.v_compress = nn.Linear(dim, self.latent_dim)
        self.out_proj = nn.Linear(self.latent_dim, dim)

    def forward(self, x, mask=None):
        B, T, D = x.shape

        q = self.q_proj(x)
        kv_raw = self.kv_proj(x).chunk(2, dim=-1)
        k_raw, v_raw = kv_raw

        k = self.k_compress(k_raw).reshape(B, T, self.heads, self.hd_latent).transpose(1, 2)
        v = self.v_compress(v_raw).reshape(B, T, self.heads, self.hd_latent).transpose(1, 2)
        q = q.reshape(B, T, self.heads, self.hd).transpose(1, 2)

        attn_score = (q @ k.transpose(-2, -1)) / (self.hd ** 0.5)
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, -1e9)

        attn_weight = F.softmax(attn_score, dim=-1)
        attn_out = (attn_weight @ v).transpose(1, 2).reshape(B, T, self.latent_dim)
        return self.out_proj(attn_out)
