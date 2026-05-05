# src/layers/moe.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseExpertMoE(nn.Module):
    """稀疏专项MoE — 256 路由专家，Top-7 激活，仅做专项补强"""

    def __init__(self, dim=8192, num_experts=256, top_k=7):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x):
        B, T, D = x.shape
        gate_logits = self.gate(x)
        top_val, top_idx = torch.topk(gate_logits, self.top_k, dim=-1)
        top_weight = F.softmax(top_val, dim=-1)

        out = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = top_idx[:, :, k]
            w = top_weight[:, :, k : k + 1]
            for e in range(self.num_experts):
                token_mask = (expert_idx == e).unsqueeze(-1).float()
                expert_out = self.experts[e](x)
                out += token_mask * w * expert_out
        return out
