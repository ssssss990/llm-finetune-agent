"""
LoRA / QLoRA 微调训练

关键参数选择理由：
- r=64：秩越大可训练参数越多，但过大容易过拟合。64 在 7B 模型上是经验最优。
- alpha=128：缩放系数，通常设为 r 的 2 倍。
- target_modules：q_proj + v_proj + down_proj
  - 走过的弯路：最初只挂 q_proj 和 v_proj，对领域术语的适配不够。
  - 补上 down_proj 后效果明显提升，因为 down_proj 负责信息压缩，
    领域术语需要在这一层被"理解"。
- QLoRA 4-bit：让 7B 模型能在单张 24G GPU 上训练

显存估算（Qwen2.5-7B + QLoRA 4-bit）：
- 模型权重 4-bit：~7GB
- LoRA 参数 fp16：~0.2GB
- 优化器状态：~0.4GB
- 激活值 + 梯度：~10GB
- 总计：~18GB（24G 卡可跑）
"""

import json
from pathlib import Path
from typing import Optional
from loguru import logger

from config.settings import settings


class LoRATrainer:
    """LoRA / QLoRA 微调器"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.peft_config = None

    def setup(self):
        """初始化模型、分词器、LoRA 配置"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType

        bnb_config = None
        if settings.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(settings.base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            settings.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        self.peft_config = LoraConfig(
            r=settings.lora_r,
            lora_alpha=settings.lora_alpha,
            target_modules=settings.lora_target_modules,
            lora_dropout=settings.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, self.peft_config)

        trainable, total = self.model.get_nb_trainable_parameters()
        logger.info(
            f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)"
        )

    def load_dataset(self, path: str):
        """加载 SFT 数据集并转换为训练格式"""
        from datasets import Dataset

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        def format_messages(sample):
            text = self.tokenizer.apply_chat_template(
                sample["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            return {"text": text}

        dataset = Dataset.from_list(data)
        dataset = dataset.map(format_messages)
        return dataset

    def train(self, train_path: str, eval_path: str = None):
        """执行 SFT 训练"""
        from trl import SFTTrainer, SFTConfig
        import torch

        self.setup()

        train_dataset = self.load_dataset(train_path)
        eval_dataset = self.load_dataset(eval_path) if eval_path else None

        training_args = SFTConfig(
            output_dir=settings.output_dir,
            num_train_epochs=settings.num_epochs,
            per_device_train_batch_size=settings.batch_size,
            gradient_accumulation_steps=settings.gradient_accumulation_steps,
            learning_rate=settings.learning_rate,
            warmup_ratio=settings.warmup_ratio,
            max_seq_length=settings.max_seq_length,
            logging_steps=10,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=100,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=3,
            bf16=True,
            gradient_checkpointing=True,
            optim="adamw_torch",
            report_to="none",
        )

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )

        logger.info("Starting SFT training...")
        trainer.train()

        # 保存 LoRA 权重
        trainer.save_model(settings.output_dir)
        self.tokenizer.save_pretrained(settings.output_dir)
        logger.info(f"Model saved to {settings.output_dir}")

    def merge_and_save(self, output_path: str):
        """
        合并 LoRA 权重到基座模型并保存

        部署时可以选择合并保存（推理快）或动态加载（灵活切换）
        """
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base = AutoModelForCausalLM.from_pretrained(
            settings.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        merged = PeftModel.from_pretrained(base, settings.output_dir)
        merged = merged.merge_and_unload()

        tokenizer = AutoTokenizer.from_pretrained(settings.output_dir)
        merged.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        logger.info(f"Merged model saved to {output_path}")
