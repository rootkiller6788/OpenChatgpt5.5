# src/layers/tool_engine.py
import torch.nn as nn


class ToolScheduler(nn.Module):
    """[Inferred] 工具调度器 — 工具选择 + 多步时序规划

    从 ChatGPT 可观察的 function calling 行为推断：GPT 支持多步骤工具调用。
    当前为 toy 架构假设，LSTM 多步调度器是纯推测性设计。
    实际工具调度更接近系统层（tool router → executor → observation → reinject）。
    """

    def __init__(self, dim=8192, tool_num=24):
        super().__init__()
        self.plan_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, tool_num),
        )
        self.step_schedule = nn.LSTM(
            dim, dim, num_layers=2, batch_first=True
        )

    def forward(self, x):
        tool_logits = self.plan_net(x)
        step_hidden, _ = self.step_schedule(x)
        return tool_logits, step_hidden
