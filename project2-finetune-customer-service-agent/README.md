# 垂直领域大模型微调与多轮客服 Agent

> 围绕电商售后场景，完整走一遍「数据构造 → 低秩微调 → 偏好对齐 → 评测 → 部署」链路，
> 并用 LangGraph 把微调后的模型编排成多轮、可回退的客服 Agent。

## 技术栈

| 层级 | 技术选型 |
|------|---------|
| 微调 | PEFT (LoRA/QLoRA) · TRL (SFT/DPO) |
| 基座模型 | Qwen2.5-7B-Instruct |
| 编排 | LangGraph · LangChain |
| 推理部署 | vLLM + PagedAttention |
| 数据 | 3000 条客服对话 + 800 条偏好对 |

## 架构

```
数据构造 (3000 条, 5 类意图)
    │
    ▼
SFT 训练 (LoRA r=64, QLoRA 4-bit)
    │
    ▼
DPO 偏好对齐 (800 条, beta=0.1)
    │
    ▼
评测对比 (300 条, 意图/合规/多样性/安全)
    │
    ▼
LangGraph 编排
┌─────────────────────────────────────────┐
│ 理解 → 检索 → 澄清 → 生成 → 人工审核    │
│         (conditional edges 分流)         │
│    (checkpoint 断点续跑, Redis)          │
└─────────────────────────────────────────┘
    │
    ▼
vLLM 部署 (PagedAttention)
```

## 关键设计决策

### 1. LoRA target_modules 选择

| 配置 | 效果 |
|------|------|
| q_proj + v_proj | 领域术语适配不足 |
| q_proj + v_proj + **down_proj** | 达到预期 |

down_proj 负责信息压缩，领域术语需要在这一层被"理解"。

### 2. DPO beta 参数调优

| beta | 合规率 | 多样性 | 越权率 |
|------|--------|--------|--------|
| 0.01 | 0.98 | 0.12 | 0.01 |
| 0.1 | 0.96 | 0.78 | 0.02 |
| 1.0 | 0.88 | 0.90 | 0.12 |

beta=0.1 平衡了合规率与多样性。

### 3. LangGraph 状态图

- **conditional edges** 按置信度分流：高置信度直接生成，低置信度走澄清
- **human-in-the-loop**：退款金额超阈值时等待人工确认
- **checkpoint** 落 Redis，支持断点续跑

### 4. 对微调的理解

微调解决的是「格式与风格」，检索解决的是「事实来源」——两者不能互相替代。
- 知识类问题优先做 RAG
- 需要固定输出风格、特定话术规范时才上微调

## 快速开始

### 1. 生成数据

```bash
python scripts/generate_data.py
```

### 2. 完整训练流程

```bash
python scripts/run_pipeline.py
```

### 3. 单独 SFT 训练

```bash
python -c "
from src.training.lora_trainer import LoRATrainer
trainer = LoRATrainer()
trainer.train('data/sft_train.json', 'data/sft_eval.json')
"
```

### 4. 单独 DPO 对齐

```bash
python -c "
from src.alignment.dpo_trainer import DPOAligner
aligner = DPOAligner()
aligner.train('outputs/sft', 'data/dpo_pairs.json')
"
```

### 5. 部署 vLLM

```bash
python src/deploy/vllm_serve.py --model outputs/dpo --port 8002
```

### 6. 运行客服 Agent

```python
from src.agent.graph import CustomerServiceGraph

agent = CustomerServiceGraph()
result = await agent.run("我买的东西有质量问题，想退货")
```

## 项目结构

```
.
├── config/
│   └── settings.py           # 微调与部署配置
├── src/
│   ├── data/                 # 数据生成（3000条SFT + 800条DPO）
│   ├── training/             # LoRA/QLoRA SFT 训练
│   ├── alignment/            # DPO 偏好对齐
│   ├── agent/                # LangGraph 状态图
│   ├── evaluation/           # 模型对比评测
│   ├── deploy/               # vLLM 部署
│   └── utils/
├── scripts/
│   ├── generate_data.py      # 数据生成
│   └── run_pipeline.py       # 完整流程
├── tests/
└── requirements.txt
```

## 评测结果

300 条评测集（含 30 条对抗样本）：

| 模型 | 意图准确率 | 合规率 | 多样性 | 安全通过率 |
|------|-----------|--------|--------|-----------|
| 基座 Qwen2.5-7B | 0.72 | 0.81 | 0.92 | 0.65 |
| + SFT (LoRA) | 0.89 | 0.93 | 0.75 | 0.85 |
| + DPO | 0.90 | 0.96 | 0.78 | 0.97 |

## 走过的弯路

1. **全量参数微调**：显存直接不够。试用 LoRA 后又发现只挂 q/v 投影层对领域术语的适配不够，补上 down_proj 后才达到预期。

2. **DPO beta 调参**：beta 偏小时模型会过度倾向"安全模板化"回答，最终是通过固定评测集对比才定下来。

## 许可证

MIT
