# src/layers/rlhf_align.py
import torch.nn as nn


class RLHFAlignmentLayer(nn.Module):
    """RLHF 对齐层 — 人类偏好 + 安全合规映射"""

    def __init__(self, dim=8192):
        super().__init__()
        self.align_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.safe_constraint = nn.Linear(dim, dim)

    def forward(self, x):
        align_feat = self.align_proj(x)
        safe_feat = self.safe_constraint(x)
        return align_feat + safe_feat
