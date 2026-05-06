# Architecture Discussion

## Important Disclaimer

OpenAI has **not** publicly disclosed GPT-4/GPT-4o/GPT-5 internal model architecture.
Layer count, hidden dimension, attention type, MoE structure — these are all unknown.

What IS known: from the GPT-4 Technical Report (dense transformer), from InstructGPT (RLHF methodology),
from ChatGPT product behavior (multimodal, function calling). This document infers **behavioral system components**
from those public signals. Implementation details are **speculative toy designs**.

---

## Evidence Hierarchy

| Tag | Source |
|-----|--------|
| `[Observed]` | API/ChatGPT behavior, black-box testing |
| `[Reported]` | OpenAI papers, blog posts, system cards |
| `[Inferred]` | Reasonably deduced system component |
| `[Speculative]` | Pure architectural hypothesis |

---

## Inferred System Components

### 1. Multimodal Frontend
`[Observed] + [Reported]`

ChatGPT accepts text + images. GPT-4V introduced a vision encoder.
The frontend must embed text tokens and project image patches into a unified hidden space.

**Toy implementation:** `MultimodalFrontend` — text_emb + image_projector + audio_projector + code_projector.
Each non-text modality projected to one special token (toy simplification, real systems use patch-level encoding).

### 2. Dense Transformer Backbone
`[Reported] + [Speculative]`

GPT-4 Technical Report confirms transformer architecture but gives zero architectural details.
Layer count, width, attention mechanism — all unknown.

**Toy implementation:** `DenseTransformerBlock` — standard Pre-LN + attention + FFN.
Stack count and dimension are arbitrary speculative choices.

### 3. Latent Compressed Attention
`[Speculative]`

KV compression (latent bottleneck) is a well-known attention optimization technique.
No evidence OpenAI uses this specific approach. The name "MLA" refers to DeepSeek's work,
not OpenAI's.

**Toy implementation:** `LatentCompressedAttention` with `latent_compress=8`.
K/V compressed via Linear projection before computing attention.

### 4. Sparse MoE
`[Speculative]`

OpenAI has not confirmed whether GPT-4 uses MoE. Various industry rumors exist but no
official confirmation. This module is entirely speculative.

**Toy implementation:** `SparseMoELayer` — naive expert computation (all experts run full forward
regardless of routing mask).

### 5. Tool Scheduler
`[Observed] + [Inferred]`

ChatGPT supports function calling with user-defined JSON schemas. The model must have
a mechanism to select appropriate tools and generate structured tool calls.

**Toy implementation:** `ToolScheduler` — plan_net for tool selection + LSTM for multi-step scheduling.
LSTM-based scheduling is a speculative design choice.

### 6. Output Alignment (RLHF)
`[Reported] + [Inferred]`

OpenAI published the RLHF methodology and confirmed its use. RLHF is a **training procedure**,
not a neural network layer that runs during inference.

**Toy implementation:** `OutputAlignmentHead` — simulates the behavioral effect of RLHF training
as a forward-pass module. This is a deliberate simplification; real RLHF occurs during training,
not as an inference-time layer.

---

## Architecture Split

Model separated into two classes:

### ToyGPTBackbone
Pure model forward: multimodal token embedding → transformer blocks → LM head.

### GPTStyleSystemRuntime
System-layer behaviors that wrap the model:
- Tool scheduling (plan + LSTM multi-step)
- Output alignment (RLHF effect simulation)

This split exists because:
1. RLHF is a training methodology, not a forward-pass layer
2. Tool calling is a system-layer concern (router → executor → observation → reinject)
3. Embedding these as nn.Module layers inside the backbone would be architecturally misleading
