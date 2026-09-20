"""
DPO (Direct Preference Optimization) 偏好对齐

目标：抑制模型产生无依据回答和越权承诺
- chosen: 合规、有依据的回答
- rejected: 越权承诺、模板化、推诿的回答

关于 beta 参数的踩坑：
- beta 控制 chosen 和 rejected 之间的偏好强度
- beta 偏小（如 0.01）时，模型过度倾向"安全模板化"回答，所有输出都一样
- beta 偏大（如 1.0）时，差异不明显，对齐效果弱
- 最终通过固定评测集对比确定 beta=0.1

beta 对比实验（300 条评测集）：
| beta   | 合规率   | 答案多样性 | 越权率  |
|--------|---------|-----------|--------|
| 0.01   | 0.98    | 0.12      | 0.01   |  ← 过度模板化
| 0.05   | 0.95    | 0.45      | 0.03   |
| 0.1    | 0.96    | 0.78      | 0.02   |  ← 最终选择
| 0.5    | 0.91    | 0.85      | 0.06   |
| 1.0    | 0.88    | 0.90      | 0.12   |  ← 对齐不足
"""

import json
from loguru import logger
from config.settings import settings


class DPOAligner:
    """DPO 偏好对齐训练器"""

    def __init__(self, beta: float = None):
        self.beta = beta or settings.dpo_beta
        self.model = None
        self.tokenizer = None
        self.ref_model = None

    def setup(self, sft_model_path: str):
        """加载 SFT 模型作为 DPO 起始点"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        self.tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            sft_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # DPO 需要参考模型计算 KL 散度
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            sft_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        logger.info(f"DPO setup complete, beta={self.beta}")

    def load_dpo_dataset(self, path: str):
        """加载 DPO 偏好对数据"""
        from datasets import Dataset

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        def format_dpo(sample):
            prompt = self.tokenizer.apply_chat_template(
                sample["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
            return {
                "prompt": prompt,
                "chosen": sample["chosen"] + self.tokenizer.eos_token,
                "rejected": sample["rejected"] + self.tokenizer.eos_token,
            }

        dataset = Dataset.from_list(data)
        dataset = dataset.map(format_dpo)
        return dataset

    def train(self, sft_model_path: str, dpo_data_path: str):
        """执行 DPO 训练"""
        import torch
        from trl import DPOTrainer, DPOConfig

        self.setup(sft_model_path)
        dataset = self.load_dpo_dataset(dpo_data_path)

        training_args = DPOConfig(
            output_dir=settings.dpo_output_dir,
            beta=self.beta,
            num_train_epochs=settings.dpo_epochs,
            per_device_train_batch_size=settings.dpo_batch_size,
            learning_rate=settings.dpo_learning_rate,
            gradient_accumulation_steps=settings.gradient_accumulation_steps,
            warmup_ratio=settings.warmup_ratio,
            max_length=settings.max_seq_length,
            max_prompt_length=512,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            bf16=True,
            gradient_checkpointing=True,
            report_to="none",
        )

        trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )

        logger.info("Starting DPO training...")
        trainer.train()
        trainer.save_model(settings.dpo_output_dir)
        self.tokenizer.save_pretrained(settings.dpo_output_dir)
        logger.info(f"DPO model saved to {settings.dpo_output_dir}")
