# src/layers/mla_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLAHiddenAttention(nn.Module):
    """MLA 多头潜注意力 — 潜空间KV压缩，保证长链逻辑一致性"""

    def __init__(self, dim=8192, heads=64, latent_compress=8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.hd = dim // heads
        self.latent_compress = latent_compress

        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.k_latent = nn.Linear(dim, dim // latent_compress)
        self.v_latent = nn.Linear(dim, dim // latent_compress)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        B, T, D = x.shape

        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = qkv

        q = q.reshape(B, T, self.heads, self.hd).transpose(1, 2)
        k = k.reshape(B, T, self.heads, self.hd).transpose(1, 2)
        v = v.reshape(B, T, self.heads, self.hd).transpose(1, 2)

        k_lat = self.k_latent(k)
        v_lat = self.v_latent(v)

        attn_score = (q @ k_lat.transpose(-2, -1)) / (self.hd**0.5)
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, -1e9)

        attn_weight = F.softmax(attn_score, dim=-1)
        attn_out = (attn_weight @ v_lat).transpose(1, 2).reshape(B, T, D)

        return self.out_proj(attn_out)
