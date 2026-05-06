# src/model.py
import torch.nn as nn

from .layers import (
    LayerNorm,
    MultimodalFrontend,
    DenseTransformerBlock,
    SparseMoELayer,
    ToolScheduler,
    OutputAlignmentHead,
)


class ToyGPTBackbone(nn.Module):
    """[Speculative] GPT 风格模型主干 — 稠密 Transformer + 可选 MoE

    从公开信息推断：GPT-4 是 dense transformer 架构（GPT-4 技术报告中提及），
    GPT-4o 支持多模态。具体的层数、维度、MoE 配置均为纯推测。
    """

    def __init__(self, vocab_size=50000, dim=8192, heads=64, num_layers=16):
        super().__init__()
        self.frontend = MultimodalFrontend(vocab_size, dim)
        self.blocks = nn.ModuleList(
            [DenseTransformerBlock(dim, heads) for _ in range(num_layers)]
        )
        self.moe = SparseMoELayer(dim)
        self.norm_out = LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, text_ids, img_feat=None, audio_feat=None, code_feat=None,
                pos_ids=None, attention_mask=None):
        x = self.frontend(text_ids, img_feat, audio_feat, code_feat, pos_ids)
        for block in self.blocks:
            x = block(x, attention_mask)
        x = x + self.moe(x)
        x = self.norm_out(x)
        logits = self.lm_head(x)
        return logits


class GPTStyleSystemRuntime(nn.Module):
    """[Inferred] GPT 风格系统运行时 — 系统层组件

    RLHF 是训练方法论，不是模型架构层。工具调用是系统层行为。
    将这些从 backbone 中分离出来，避免暗示它们是 Transformer 内部层。
    """

    def __init__(self, dim=8192, tool_num=24):
        super().__init__()
        self.tool = ToolScheduler(dim, tool_num)
        self.align = OutputAlignmentHead(dim)

    def forward(self, x):
        tool_logits, step_hidden = self.tool(x)
        x = x + step_hidden
        x = self.align(x)
        return x, tool_logits
