# src/layers/embedding.py
import torch
import torch.nn as nn


class MultimodalFrontend(nn.Module):
    """[Speculative] 多模态前端 — 文本 embedding + 模态投影 → 拼接为统一序列

    每个模态单独投影到统一 hidden dimension，作为独立 token 拼接到文本序列上。
    当前为 toy 实现，模态 token 各用一个 token 表示（非 patch 级编码）。
    没有证据表明 OpenAI 使用这种特定的融合方式。
    """

    def __init__(self, vocab_size=50000, dim=8192):
        super().__init__()
        self.dim = dim
        self.text_emb = nn.Embedding(vocab_size, dim)
        self.image_projector = nn.Sequential(
            nn.Linear(2048, dim), nn.GELU(), nn.Linear(dim, dim),
        )
        self.audio_projector = nn.Sequential(
            nn.Linear(1024, dim), nn.GELU(), nn.Linear(dim, dim),
        )
        self.code_projector = nn.Sequential(
            nn.Linear(4096, dim), nn.GELU(), nn.Linear(dim, dim),
        )
        self.type_emb = nn.Embedding(4, dim)
        self.pos_emb = nn.Embedding(262144, dim)

    def forward(self, text_ids, img_feat=None, audio_feat=None, code_feat=None, pos_ids=None):
        text_x = self.text_emb(text_ids)
        parts = [text_x]
        type_t = torch.zeros(text_ids.shape[0], text_ids.shape[1], dtype=torch.long, device=text_ids.device)

        if img_feat is not None:
            parts.append(self.image_projector(img_feat).unsqueeze(1))
            type_t = torch.cat([type_t, torch.ones(text_ids.shape[0], 1, dtype=torch.long, device=text_ids.device)], dim=1)
        if audio_feat is not None:
            parts.append(self.audio_projector(audio_feat).unsqueeze(1))
            type_t = torch.cat([type_t, torch.full((text_ids.shape[0], 1), 2, dtype=torch.long, device=text_ids.device)], dim=1)
        if code_feat is not None:
            parts.append(self.code_projector(code_feat).unsqueeze(1))
            type_t = torch.cat([type_t, torch.full((text_ids.shape[0], 1), 3, dtype=torch.long, device=text_ids.device)], dim=1)

        x = torch.cat(parts, dim=1)
        x = x + self.type_emb(type_t)
        if pos_ids is not None:
            x = x + self.pos_emb(pos_ids)
        return x
