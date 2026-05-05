# GPT-5.5 Architecture Inference

GPT-5.5 完整主模型 — 稠密 Transformer 主干 + 稀疏 MoE 专项补强的架构推理实现，用于学习和理解模型结构。

---

## 项目结构

```
openchatgpt/
├── src/
│   ├── __init__.py
│   ├── model.py                  # GPT-5.5 完整主模型
│   └── layers/
│       ├── __init__.py           # 子模块统一导出
│       ├── common.py             # LayerNorm 基础组件
│       ├── embedding.py          # 早期融合嵌入（文本/图像/音频/代码）
│       ├── mla_attention.py      # MLA 多头潜注意力
│       ├── dense_block.py        # 稠密FFN + DenseTransformerBlock
│       ├── moe.py                # 稀疏专项MoE（256专家）
│       ├── tool_engine.py        # 原生工具调度引擎
│       └── rlhf_align.py         # RLHF对齐层
├── demo/
│   └── infer_demo.py             # 推理演示入口
├── docs/
│   ├── ARCHITECTURE.md           # 架构详解（含三模型对比）
│   └── ascii_arch.txt            # ASCII 架构图
├── assets/                       # 架构图（待补充）
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## 如何阅读这个项目

### 推荐阅读顺序（自底向上）

**第一步：基础组件** — `src/layers/common.py`
- `LayerNorm`，三个模型统一复用。

**第二步：嵌入层** — `src/layers/embedding.py`
- `MultimodalEarlyFusionEmbedding`：文本/图像/音频/代码**四路各自投影到 8192 维**，然后 concat 到 `dim*4`，再用 `fusion_merge` 压回 `dim`，最后叠加位置编码。
- 这是 GPT-5.5 最独特的嵌入设计——**早期融合**（Early Fusion），在嵌入阶段就把所有模态的信息揉在一起，而不是像 Gemini 那样在序列维度拼接后置处理。
- 阅读关键：注意 `repeat(1, T_txt, 1)` 把单向量模态特征广播到整个文本序列长度，实现逐 token 融合。

**第三步：MLA 注意力** — `src/layers/mla_attention.py`
- `MLAHiddenAttention`：MLA = Multi-head Latent Attention。核心特���是 **KV 潜空间压缩**——在投影完 Q/K/V 后，额外对 K 和 V 做一次 bottleneck 压缩（`dim → dim/8`），在压缩后的潜空间里计算注意力，降低计算量同时保留长程逻辑一致性。
- 对比：Gemini 用 Conv1d 压缩 KV（CSA/HCA），GPT-5.5 用线性 projection 压缩，都是降计算但手段不同。

**第四步：稠密主干** — `src/layers/dense_block.py`
- `DenseSharedFFN`：标准的 `Linear → GELU → Linear`，但它是 **共享的**，承载约 45% 的总参数量，负责通用推理能力。
- `DenseTransformerBlock`：Pre-LN → MLA → 残差 → Pre-LN → DenseSharedFFN → 残差。就是一个标准 Pre-Norm Transformer 块。
- GPT-5.5 堆叠 **48 层** 这种同构块构成主干。对比：Claude 只有 1 个块循环 12 次，Gemini 有 40 个 CSA/HCA 交替块。

**第五步：MoE 与工具** — `src/layers/moe.py` → `src/layers/tool_engine.py`
- `SparseExpertMoE`：256 专家，top-7 路由。关键定位：**不参与通用语义建模，只做代码/数学/工具专项补强**。这与 Claude/Gemini 的 MoE 承担全部 token 推理不同。
- `NativeToolScheduler`：双阶段 —— (1) `plan_net` 输出 24 种工具的选择概率，(2) `LSTM` 做多步骤时序调度，输出隐状态叠加回主表示。全自主调度，对比 Claude 的被动工具层。

**第六步：RLHF 对齐** — `src/layers/rlhf_align.py`
- `RLHFAlignmentLayer`：alignment projection + safety constraint，两者相加作为最终对齐输出。这是 GPT-5.5 的安全机制，对比 Claude 的前置+后置宪法AI双校验。

**第七步：串联全部** — `src/model.py`
- `GPT55FullArch`：嵌入 → 48 层稠密主干 → MoE 残差补强 → 工具调度 → RLHF 对齐 → 归一化 → LM 头。

---

## 数据流全景

```
输入（text_ids, img_feat, audio_feat, code_feat, pos_ids）
  │
  ▼
MultimodalEarlyFusionEmbedding   —— 四模态各自投影 → concat → fusion merge + pos
  │
  ▼
┌─ DenseTransformerBlock × 48 ──┐
│   ├─ Pre-LN                     │
│   ├─ MLAHiddenAttention          │  ← KV 潜空间压缩
│   ├─ Pre-LN                     │
│   └─ DenseSharedFFN             │  ← 占45%参数
└────────────────────────────────┘
  │
  ▼
SparseExpertMoE                 —— 256 专家，top-7（仅专项补强）
  │
  ▼
NativeToolScheduler             —— 工具选择 + LSTM 多步骤调度
  │
  ▼
RLHFAlignmentLayer              —— 偏好对齐 + 安全约束
  │
  ▼
LayerNorm → LM Head             —— 最终归一化 → logits 输出
```

---

## 安装与运行

```bash
# 创建并激活 venv
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/macOS

# 安装依赖
pip install -r requirements.txt
```

```bash
# 运行推理演示（检查模型能否正常 forward）
python demo/infer_demo.py
```

---

## 关键参数速查

| 参数 | 值 | 说明 |
|------|-----|------|
| `dim` | 8192 | 模型隐层维度 |
| `heads` | 64 | 注意力头数 |
| `head_dim` | 128 | 每头维度 (8192/64) |
| `layer_num` | 48 | 稠密 Transformer 块堆叠数 |
| `vocab_size` | 50000 | 词表大小 |
| `num_experts` | 256 | MoE 路由专家数 |
| `top_k (MoE)` | 7 | 每 token 激活专家数 |
| `latent_compress` | 8 | MLA 潜空间 KV 压缩比 |
| `mlp_ratio` | 4 | DenseSharedFFN 扩展比 |
| `max_pos` | 262144 | 最大位置编码长度 |
| `tool_num` | 24 | 工具调度引擎支持的工具数 |
| `code_dim` | 4096 | 代码模态输入维度 |
| `img_dim` | 2048 | 图像模态输入维度 |
| `audio_dim` | 1024 | 音频模态输入维度 |

---

## 三模型架构全景对比

| 维度 | GPT-5.5 | Claude Mythos | Gemini 3.1 Pro |
|------|---------|--------------|----------------|
| **深度方式** | 48 层堆叠 | RDT 循环（权重复用 12 轮） | 40 层 CSA+HCA 交替 |
| **注意力** | MLA（潜空间 KV 压缩） | 局部窗口（128） | CSA(4×)+HCA(128×) |
| **MoE 定位** | 仅专项补强 | 全部 token | 全部 token |
| **MoE 规模** | 256 专家 / top-7 | 64 专家 / top-2 | 256 专家 / top-8 |
| **模态融合** | 早期融合（embedding 阶段） | Type embeddings | 后置拼接 |
| **安全机制** | RLHF 对齐层 | 宪法AI（前+后） | 无 |
| **工具策略** | 自主调度（LSTM 多步） | 被动（用户指令门控） | 自主（规划→校验） |
| **维度** | 8192 | 4096 | 8192 |
| **总层数** | 48 | 1 (循环12轮) | 40 |
