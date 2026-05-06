# src/layers/moe.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseMoELayer(nn.Module):
    """[Speculative] 稀疏 MoE 层 — N 路由专家 top-k 激活

    MoE 是行业常用技术，但 OpenAI 未公开 GPT-4/5 是否使用 MoE 或具体结构。
    当前为 naive 参考实现：每个 expert 对完整 batch 执行前向计算，
    即使 mask 为空也算一遍，未做负载均衡。
    """

    def __init__(self, dim=8192, num_experts=256, top_k=8):
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
            w = top_weight[:, :, k: k + 1]
            for e in range(self.num_experts):
                token_mask = (expert_idx == e).unsqueeze(-1).float()
                expert_out = self.experts[e](x)
                out += token_mask * w * expert_out
        return out
