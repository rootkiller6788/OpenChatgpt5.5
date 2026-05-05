# src/layers/embedding.py
import torch
import torch.nn as nn


class MultimodalEarlyFusionEmbedding(nn.Module):
    """早期融合嵌入层 — 文本/图像/音频/代码四路投影 → 共享Token空间融合"""

    def __init__(self, vocab_size=50000, dim=8192):
        super().__init__()
        self.dim = dim

        self.text_emb = nn.Embedding(vocab_size, dim)
        self.img_proj = nn.Sequential(
            nn.Linear(2048, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(1024, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.code_proj = nn.Sequential(
            nn.Linear(4096, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.fusion_merge = nn.Linear(dim * 4, dim)
        self.pos_emb = nn.Embedding(262144, dim)

    def forward(self, text_ids, img_feat, audio_feat, code_feat, pos_ids):
        B, T_txt = text_ids.shape

        t_emb = self.text_emb(text_ids)
        i_emb = self.img_proj(img_feat).unsqueeze(1).repeat(1, T_txt, 1)
        a_emb = self.audio_proj(audio_feat).unsqueeze(1).repeat(1, T_txt, 1)
        c_emb = self.code_proj(code_feat).unsqueeze(1).repeat(1, T_txt, 1)

        concat = torch.cat([t_emb, i_emb, a_emb, c_emb], dim=-1)
        fused = self.fusion_merge(concat)
        fused = fused + self.pos_emb(pos_ids)
        return fused
