# demo/infer_demo.py
"""GPT-5.5 推理演示"""

import torch
from src.model import GPT55FullArch


def main():
    model = GPT55FullArch(
        vocab_size=50000,
        dim=8192,
        heads=64,
        layer_num=48,
    )
    model.eval()

    B = 1
    T = 512
    text_ids = torch.randint(0, 50000, (B, T))
    img_feat = torch.randn(B, 2048)
    audio_feat = torch.randn(B, 1024)
    code_feat = torch.randn(B, 4096)
    pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1)

    print("Model loaded. Running forward pass...")
    with torch.no_grad():
        logits, tool_logits = model(
            text_ids, img_feat, audio_feat, code_feat, pos_ids
        )
    print(f"Output logits shape: {logits.shape}")
    print(f"Tool logits shape: {tool_logits.shape}")


if __name__ == "__main__":
    main()
