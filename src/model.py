# src/model.py
import torch.nn as nn

from .layers import (
    LayerNorm,
    MultimodalEarlyFusionEmbedding,
    DenseTransformerBlock,
    SparseExpertMoE,
    NativeToolScheduler,
    RLHFAlignmentLayer,
)


class GPT55FullArch(nn.Module):
    """GPT-5.5 完整主模型 — 所有层串死，和架构图 1:1"""

    def __init__(self, vocab_size=50000, dim=8192, heads=64, layer_num=48):
        super().__init__()

        self.embedding = MultimodalEarlyFusionEmbedding(vocab_size, dim)

        self.dense_blocks = nn.ModuleList(
            [DenseTransformerBlock(dim, heads) for _ in range(layer_num)]
        )

        self.moe = SparseExpertMoE(dim, num_experts=256, top_k=7)
        self.tool_engine = NativeToolScheduler(dim)
        self.rlhf_align = RLHFAlignmentLayer(dim)

        self.norm_out = LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(
        self, text_ids, img_feat, audio_feat, code_feat, pos_ids, attention_mask=None
    ):
        x = self.embedding(text_ids, img_feat, audio_feat, code_feat, pos_ids)

        for block in self.dense_blocks:
            x = block(x, attention_mask)

        x = x + self.moe(x)

        tool_logits, step_hidden = self.tool_engine(x)
        x = x + step_hidden

        x = self.rlhf_align(x)

        x = self.norm_out(x)
        logits = self.lm_head(x)

        return logits, tool_logits
