# GPT-5.5 Architecture Inference

Architecture inference of GPT-5.5 — a dense Transformer backbone with sparse MoE
specialization. For learning and understanding the model structure.

---

## Project Structure

```
openchatgpt/
├── src/
│   ├── __init__.py
│   ├── model.py                  # GPT-5.5 full main model
│   └── layers/
│       ├── __init__.py           # Unified exports
│       ├── common.py             # LayerNorm base component
│       ├── embedding.py          # Early fusion embedding (text/image/audio/code)
│       ├── mla_attention.py      # MLA Multi-head Latent Attention
│       ├── dense_block.py        # DenseSharedFFN + DenseTransformerBlock
│       ├── moe.py                # Sparse Expert MoE (256 experts)
│       ├── tool_engine.py        # Native tool scheduling engine
│       └── rlhf_align.py         # RLHF alignment layer
├── demo/
│   └── infer_demo.py             # Inference demo entry point
├── docs/
│   ├── ARCHITECTURE.md           # Architecture deep-dive (with 3-model comparison)
│   └── ascii_arch.txt            # ASCII architecture diagram
├── assets/                       # Architecture diagram (TBD)
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## How to Read This Project

### Recommended Reading Order (Bottom-Up)

**Step 1: Base Components** — `src/layers/common.py`
- `LayerNorm`: shared across all three models.

**Step 2: Embedding Layer** — `src/layers/embedding.py`
- `MultimodalEarlyFusionEmbedding`: Each of the four modalities (text/image/audio/code) is independently projected to 8192 dims, concatenated to `dim*4`, then merged back to `dim` via `fusion_merge`, and finally position embeddings are added.
- This is GPT-5.5's most distinctive embedding design — **Early Fusion** fuses all modal information at the embedding stage, unlike Gemini's post-concatenation approach.
- Key insight: Note how `repeat(1, T_txt, 1)` broadcasts single-vector modality features across the full text sequence for per-token fusion.

**Step 3: MLA Attention** — `src/layers/mla_attention.py`
- `MLAHiddenAttention`: MLA = Multi-head Latent Attention. Core feature is **KV latent compression** — after Q/K/V projection, an additional bottleneck compression (`dim → dim/8`) is applied to K and V. Attention is computed in the compressed latent space, reducing compute while preserving long-range logical consistency.
- Compare: Gemini uses Conv1d to compress KV (CSA/HCA); GPT-5.5 uses linear projection — different mechanisms, same goal.

**Step 4: Dense Backbone** — `src/layers/dense_block.py`
- `DenseSharedFFN`: Standard `Linear → GELU → Linear`, but **shared** — it carries ~45% of total parameters and handles general reasoning capability.
- `DenseTransformerBlock`: Pre-LN → MLA → residual → Pre-LN → DenseSharedFFN → residual. A standard Pre-Norm Transformer block.
- GPT-5.5 stacks **48 layers** of these homogeneous blocks to form the backbone. Compare: Claude has only 1 block looped 12 times; Gemini has 40 alternating CSA/HCA blocks.

**Step 5: MoE and Tools** — `src/layers/moe.py` → `src/layers/tool_engine.py`
- `SparseExpertMoE`: 256 experts, top-7 routing. Key positioning: **does not participate in general semantic modeling — only reinforces code/math/tool specialization**. This differs from Claude/Gemini where MoE handles all token inference.
- `NativeToolScheduler`: Two-stage — (1) `plan_net` outputs selection probabilities for 24 tools, (2) `LSTM` performs multi-step temporal scheduling, with hidden states fused back into the main representation. Fully autonomous, unlike Claude's passive tool layer.

**Step 6: RLHF Alignment** — `src/layers/rlhf_align.py`
- `RLHFAlignmentLayer`: Alignment projection + safety constraint mask, added together as the final aligned output. This is GPT-5.5's safety mechanism, compared to Claude's pre+post Constitutional AI dual check.

**Step 7: Assemble Everything** — `src/model.py`
- `GPT55FullArch`: Embedding → 48-layer dense backbone → MoE residual reinforcement → tool scheduling → RLHF alignment → norm → LM head.

---

## Data Flow

```
Input (text_ids, img_feat, audio_feat, code_feat, pos_ids)
  │
  ▼
MultimodalEarlyFusionEmbedding   —— four modalities individually projected → concat → fusion merge + pos
  │
  ▼
┌─ DenseTransformerBlock × 48 ──┐
│   ├─ Pre-LN                     │
│   ├─ MLAHiddenAttention          │  ← KV latent compression
│   ├─ Pre-LN                     │
│   └─ DenseSharedFFN             │  ← ~45% of total params
└────────────────────────────────┘
  │
  ▼
SparseExpertMoE                 —— 256 experts, top-7 (specialized only)
  │
  ▼
NativeToolScheduler             —— tool selection + LSTM multi-step scheduling
  │
  ▼
RLHFAlignmentLayer              —— preference alignment + safety constraint
  │
  ▼
LayerNorm → LM Head             —— final norm → logits output
```

---

## Setup and Run

```bash
# Create and activate venv
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

```bash
# Run inference demo (verify model forward pass)
python demo/infer_demo.py
```

---

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `dim` | 8192 | Hidden dimension |
| `heads` | 64 | Attention heads |
| `head_dim` | 128 | Per-head dimension (8192/64) |
| `layer_num` | 48 | Dense Transformer block stack count |
| `vocab_size` | 50000 | Vocabulary size |
| `num_experts` | 256 | Number of routed experts |
| `top_k (MoE)` | 7 | Tokens activated per expert |
| `latent_compress` | 8 | MLA KV latent compression ratio |
| `mlp_ratio` | 4 | DenseSharedFFN expansion ratio |
| `max_pos` | 262144 | Maximum position embedding length |
| `tool_num` | 24 | Number of supported tools |
| `code_dim` | 4096 | Code modality input dimension |
| `img_dim` | 2048 | Image modality input dimension |
| `audio_dim` | 1024 | Audio modality input dimension |

---

## Three-Model Architecture Comparison

| Aspect | GPT-5.5 | Claude Mythos | Gemini 3.1 Pro |
|--------|---------|--------------|----------------|
| **Depth approach** | 48-layer stack | RDT loop (weight reuse ×12) | 40-layer CSA+HCA alternating |
| **Attention** | MLA (latent KV compression) | Local window (128) | CSA(4×)+HCA(128×) |
| **MoE role** | Specialized only | All tokens | All tokens |
| **MoE scale** | 256 experts / top-7 | 64 experts / top-2 | 256 experts / top-8 |
| **Modal fusion** | Early fusion (embedding stage) | Type embeddings | Post-embedding concat |
| **Safety** | RLHF alignment layer | Constitutional AI (pre+post) | None |
| **Tool strategy** | Autonomous scheduler (LSTM) | Passive (user-gated) | Autonomous (plan→verify) |
| **Dimension** | 8192 | 4096 | 8192 |
| **Total layers** | 48 | 1 (looped 12 times) | 40 |
