# src/layers/rlhf_align.py
import torch.nn as nn


class OutputAlignmentHead(nn.Module):
    """[Inferred] 输出对齐头 — 偏好对齐 + 安全约束

    RLHF 是一种训练方法论，不是一个可以在 forward 中执行的神经网络层。
    这个模块模拟的是 RLHF 训练后产生的行为效果（对齐、安全约束），
    而非实际的 RLHF 训练过程本身。

    [Reported] OpenAI 公开使用 RLHF 进行对齐训练。
    [Speculative] 将 RLHF 效果模拟为一个 nn.Module 层是纯架构假设。
    """

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
