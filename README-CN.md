# GPT 风格稠密 Transformer 架构假设

**从公开可观察信号推断 GPT 的系统级行为架构。**

这不是对 GPT-4/GPT-4o 内部权重或源码的逆向工程。
OpenAI 未公开 GPT-4、GPT-4o 及后续模型的内部架构（层数、注意力类型、MoE 结构、隐藏维度等均未公开）。
公开可知的是 GPT 的**能力、产品机制、工具使用行为、多模态支持、系统级交互模式** —— 可通过 ChatGPT 产品、API 文档和 OpenAI 发表的研究观察到。

本项目从可观察行为中推断**行为架构**。具体实现细节（层数、MoE 配置、KV 压缩比等）均为**推测性质玩具实现**，已明确标注。

---

## 证据层级

| 标签 | 含义 | 示例 |
|------|------|------|
| `[Observed]` | API/ChatGPT 行为、黑盒测试可直接观察 | Function calling 返回结构化 JSON；ChatGPT 接受图文输入 |
| `[Reported]` | OpenAI 论文、博客、系统卡 | GPT-4 技术报告提到 dense transformer；InstructGPT 论文描述 RLHF 方法 |
| `[Inferred]` | 从观察/报告行为中合理推导 | 必须存在工具路由器来派发 function call；RLHF 训练产生对齐行为 |
| `[Speculative]` | 纯架构假设 / toy 实现 | 具体层数、KV 压缩比、MoE 专家数 |

---

## GPT 的公开信息

### 来自 OpenAI 发表的文献

**[Reported]** 来自官方来源：
- **GPT-4 技术报告**：确认 GPT-4 是 transformer 架构，但未公开任何架构细节
- **InstructGPT / RLHF**：OpenAI 发表了 RLHF 训练方法论 (`arxiv:2203.02155`)
- **GPT-4V**：多模态视觉编码器（已宣布但架构未公开）

**[Observed]** 通过 ChatGPT 和 API：
- Function calling 结构化 JSON 调用
- ChatGPT 图文多模态输入
- 长上下文支持（128K）
- 结构化输出（JSON 模式）
- 工具/插件编排

---

## 推断的系统级架构

```
用户输入
    │
    ▼
[Inferred]  多模态前端
[Observed]  ChatGPT 接受文本 + 图像
[Reported]  GPT-4V 使用视觉编码器 + 文本分词器
    │
    ├─ 文本分词器 / embedding
    ├─ 图像编码器 / patch projector
    └─（代码/音频为推测）
    │
    ▼
[Reported]  稠密 Transformer 主干
[Reported]  GPT-4 技术报告确认 transformer 架构
[Speculative] 具体深度、宽度、注意力机制
[Speculative] KV 潜空间压缩
    │
    ▼
[Speculative] 可选稀疏 MoE FFN
[Speculative] 专家数、路由方式未知
    │
    ▼
[Observed]  工具使用运行时
[Observed]  Function calling 结构化 schema
[Inferred]  工具规划/调度器 + 执行器
    │
    ▼
[Reported]  RLHF 对齐
[Reported]  RLHF 用于训练（InstructGPT 论文）
[Observed]  对齐、安全意识的输出
    │
    ▼
输出
```

---

## 组件证据映射

| 组件 | 文件 | 证据 | 理由 |
|------|------|------|------|
| 多模态前端 | `embedding.py` | [Inferred] + [Observed] | ChatGPT 接受图文；GPT-4V 视觉编码器已报道 |
| 潜空间压缩注意力 | `mla_attention.py` | [Speculative] | KV 压缩是常见优化；无证据 OpenAI 使用此形式 |
| 稠密 Transformer 块 | `dense_block.py` | [Inferred] + [Reported] | GPT-4 技术报告确认 transformer；具体架构未知 |
| 稀疏 MoE 层 | `moe.py` | [Speculative] | MoE 是行业常用；GPT-4 MoE 未经 OpenAI 确认 |
| 工具调度器 | `tool_engine.py` | [Inferred] + [Observed] | Function calling 可观察；调度机制为推断 |
| 输出对齐头 | `rlhf_align.py` | [Reported] + [Inferred] | RLHF 是已发表方法论；将对齐模拟为层是推测性的 |
| LayerNorm | `common.py` | [Inferred] | Transformer 架构中普遍存在 |

---

## 架构拆分

### ToyGPTBackbone（纯模型前向）
```
MultimodalFrontend → [DenseTransformerBlock × N] → SparseMoE → Norm → LM Head
```
text/image/audio/code → hidden → logits。所有具体选择 `[Speculative]`。

### GPTStyleSystemRuntime（系统层）
```
ToolScheduler → OutputAlignmentHead
```
工具规划 + RLHF 介导的对齐。RLHF 是训练方法论不是架构层；
此模块模拟行为效果而非训练过程本身。

---

## 本项目不是什么

- **不是** GPT-4/GPT-4o 的泄露或逆向源码
- **不是** OpenAI 训练管线或权重的复现
- **不是** 声称 GPT 使用特定层数、MoE 配置或 KV 压缩
- **不是** 生产模型 — 纯教育性玩具实现

---

## 安装与运行

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

```bash
python demo/infer_demo.py
```

---

## 参考文献

### OpenAI 来源（公开）
- GPT-4 Technical Report (`arxiv:2303.08774`)
- Training language models to follow instructions with human feedback (`arxiv:2203.02155`)
- GPT-4V(ision) System Card
- OpenAI API 文档
- ChatGPT 产品行为
