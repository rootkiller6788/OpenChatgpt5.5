# src/layers/tool_engine.py
import torch.nn as nn


class NativeToolScheduler(nn.Module):
    """原生工具调度引擎 — 多步骤规划 / 跨软件自主执行"""

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
