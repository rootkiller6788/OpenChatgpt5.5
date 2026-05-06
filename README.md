# GPT-style Dense Transformer Architecture Hypothesis

**Inferring GPT's system-level behavioral architecture from publicly observable signals.**

This is **not** a reverse-engineering of GPT-4/GPT-4o/GPT-5 internal weights or source code.
OpenAI has not publicly disclosed the internal model architecture of GPT-4, GPT-4o, or any subsequent
model (layer count, attention type, MoE structure, hidden dimension, etc.).
What IS publicly known are GPT's **capabilities, product mechanisms, tool-use behavior, multimodal support,
and system-level interaction patterns** — observable through the ChatGPT product, API documentation, and OpenAI's published research.

This project infers a **behavioral architecture** from those observables. Implementation details
(specific layer counts, MoE structure, KV compression ratio) are **speculative toy implementations**
labeled accordingly.

---

## Evidence Hierarchy

| Tag | Meaning | Example Source |
|-----|---------|---------------|
| `[Observed]` | Directly observable through API/ChatGPT behavior and black-box testing | Function calling returns structured JSON; multimodal input accepted in ChatGPT |
| `[Reported]` | Stated in OpenAI papers, blog posts, system cards | GPT-4 Technical Report mentions dense transformer; RLHF usage described in InstructGPT paper |
| `[Inferred]` | Reasonably deduced system components from observed/reported behaviors | A tool router must exist to dispatch function calls; RLHF training produces aligned behavior |
| `[Speculative]` | Pure architectural hypothesis / toy implementation | Specific layer count, KV compression ratio, MoE expert count |

---

## What Is Publicly Known About GPT

### From OpenAI's Published Work

**[Reported]** from official sources:
- **GPT-4 Technical Report**: "GPT-4 is a Transformer-style model pre-trained to predict the next token... using publicly available data and data licensed from third-party providers." No architectural details disclosed.
- **InstructGPT / RLHF**: OpenAI published the RLHF training methodology (`arxiv:2203.02155`).
- **GPT-4V**: Multimodal support via vision encoder (publicly announced but architecture undisclosed).

**[Observed]** through ChatGPT and API:
- Function calling with structured JSON schemas
- Multimodal input (text + image) in ChatGPT
- Long context support (128K)
- Structured output (JSON mode)
- Tool/plugin orchestration in ChatGPT

---

## Inferred System-Level Architecture

Based on observable behaviors and publicly reported information:

```
User Input
    │
    ▼
[Inferred]  Multimodal Frontend
[Observed]  Accepts text + image in ChatGPT
[Reported]  GPT-4V uses vision encoder + text tokenizer
    │
    ├─ Text tokenizer / embedding
    ├─ Image encoder / patch projector
    └─ (Code/Audio speculative)
    │
    ▼
[Reported]  Dense Transformer Backbone
[Reported]  GPT-4 Technical Report confirms transformer architecture
[Speculative] Specific depth, width, attention mechanism
[Speculative] Latent KV compression for efficiency
    │
    ▼
[Speculative] Optional Sparse MoE FFN
[Speculative] Expert count, routing mechanism unknown
    │
    ▼
[Observed]  Tool-use Runtime
[Observed]  Function calling with structured schemas
[Inferred]  Tool planner/scheduler + executor
    │
    ▼
[Reported]  RLHF Alignment
[Reported]  RLHF used in training (InstructGPT)
[Observed]  Aligned, safety-conscious outputs
    │
    ▼
Output
```

---

## Project Structure

```
openchatgpt/
├── src/
│   ├── __init__.py
│   ├── model.py                        # ToyGPTBackbone + GPTStyleSystemRuntime
│   └── layers/
│       ├── __init__.py
│       ├── common.py                   # [Inferred] LayerNorm
│       ├── embedding.py                # [Speculative] MultimodalFrontend
│       ├── mla_attention.py            # [Speculative] LatentCompressedAttention
│       ├── dense_block.py              # [Speculative] DenseFFN + DenseTransformerBlock
│       ├── moe.py                      # [Speculative] SparseMoELayer (naive ref impl)
│       ├── tool_engine.py              # [Inferred] ToolScheduler
│       └── rlhf_align.py               # [Inferred] OutputAlignmentHead
├── demo/
│   └── infer_demo.py                   # Forward pass verification
├── docs/
│   ├── ARCHITECTURE.md
│   └── ascii_arch.txt
├── assets/
├── requirements.txt
├── .gitignore
├── LICENSE
└── openchatgpt.txt
```

---

## Component Evidence Mapping

| Component | File | Evidence | Rationale |
|-----------|------|----------|-----------|
| Multimodal Frontend | `embedding.py` | [Inferred] + [Observed] | ChatGPT accepts text+image; GPT-4V vision encoder reported |
| Latent Compressed Attention | `mla_attention.py` | [Speculative] | KV compression is a common optimization; no evidence OpenAI uses this specific form |
| Dense Transformer Block | `dense_block.py` | [Inferred] + [Reported] | GPT-4 Technical Report confirms transformer; specific architecture unknown |
| Sparse MoE Layer | `moe.py` | [Speculative] | MoE is industry-common; GPT-4 MoE usage unconfirmed by OpenAI |
| Tool Scheduler | `tool_engine.py` | [Inferred] + [Observed] | Function calling observed; scheduling mechanism inferred |
| Output Alignment Head | `rlhf_align.py` | [Reported] + [Inferred] | RLHF is published methodology; simulating alignment as a head is speculative |
| LayerNorm | `common.py` | [Inferred] | Universally present in transformer architectures |

---

## Architecture Split

### ToyGPTBackbone (pure model forward)
```
MultimodalFrontend → [DenseTransformerBlock × N] → SparseMoE → Norm → LM Head
```
Text/image/audio/code → hidden → logits. All specific choices `[Speculative]`.

### GPTStyleSystemRuntime (system-layer)
```
ToolScheduler → OutputAlignmentHead
```
Tool planning + RLHF-mediated alignment. RLHF is a training methodology, not an architecture layer;
this module simulates the behavioral effect, not the training process itself.

---

## What This Project Is NOT

- **NOT** a leaked or reverse-engineered source of GPT-4/GPT-4o
- **NOT** a reproduction of OpenAI's training pipeline or weights
- **NOT** a claim that GPT uses specific layer counts, MoE configurations, or KV compression
- **NOT** a production model — purely speculative research implementations

---

## Setup and Run

```bash
# Create and activate venv
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

```bash
python demo/infer_demo.py
```

---

## References

### OpenAI Sources (public)
- GPT-4 Technical Report (`arxiv:2303.08774`)
- Training language models to follow instructions with human feedback / InstructGPT (`arxiv:2203.02155`)
- GPT-4V(ision) System Card
- OpenAI API documentation
- ChatGPT product behavior
