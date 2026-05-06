# demo/infer_demo.py
"""GPT-style architecture inference demo"""

import torch
from src.model import ToyGPTBackbone, GPTStyleSystemRuntime


def main():
    dim = 8192
    backbone = ToyGPTBackbone(
        vocab_size=50000,
        dim=dim,
        heads=64,
        num_layers=2,  # small for demo
    )
    agent = GPTStyleSystemRuntime(dim, tool_num=24)

    backbone.eval()
    agent.eval()

    B, T = 1, 64
    text_ids = torch.randint(0, 50000, (B, T))
    img_feat = torch.randn(B, 2048)
    audio_feat = torch.randn(B, 1024)
    code_feat = torch.randn(B, 4096)

    total_t = T + 3  # text + image + audio + code
    pos_ids = torch.arange(total_t).unsqueeze(0).expand(B, -1)

    print("Running backbone forward pass...")
    with torch.no_grad():
        logits = backbone(text_ids, img_feat, audio_feat, code_feat, pos_ids)
    print(f"Backbone logits shape: {logits.shape}")

    print("Running agent runtime forward pass...")
    with torch.no_grad():
        x_hidden = torch.randn(1, total_t, dim)
        x_out, tool_logits = agent(x_hidden)
    print(f"Agent output shape: {x_out.shape}")
    print(f"Tool logits shape: {tool_logits.shape}")


if __name__ == "__main__":
    main()
