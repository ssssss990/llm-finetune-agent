"""
完整训练流程脚本

用法：
python scripts/run_pipeline.py

执行：
1. 生成数据
2. SFT 训练（LoRA + QLoRA）
3. DPO 对齐
4. 评测对比
"""

import asyncio
import json
from loguru import logger

from config.settings import settings


async def run_pipeline():
    # Step 1: 生成数据
    logger.info("=" * 60)
    logger.info("Step 1: Generating data")
    logger.info("=" * 60)
    from src.data.data_generator import CustomerServiceDataGenerator
    gen = CustomerServiceDataGenerator()
    samples = gen.generate(3000)
    gen.save_to_sft_format(samples[:2700], "data/sft_train.json")
    gen.save_to_sft_format(samples[2700:], "data/sft_eval.json")
    dpo_pairs = gen.generate_dpo_pairs(samples, 800)
    gen.save_dpo_pairs(dpo_pairs, "data/dpo_pairs.json")

    # Step 2: SFT 训练
    logger.info("=" * 60)
    logger.info("Step 2: SFT training (LoRA + QLoRA)")
    logger.info(f"  r={settings.lora_r}, alpha={settings.lora_alpha}")
    logger.info(f"  target_modules={settings.lora_target_modules}")
    logger.info(f"  epochs={settings.num_epochs}, lr={settings.learning_rate}")
    logger.info("=" * 60)
    from src.training.lora_trainer import LoRATrainer
    trainer = LoRATrainer()
    trainer.train("data/sft_train.json", "data/sft_eval.json")

    # Step 3: DPO 对齐
    logger.info("=" * 60)
    logger.info("Step 3: DPO alignment")
    logger.info(f"  beta={settings.dpo_beta}")
    logger.info("=" * 60)
    from src.alignment.dpo_trainer import DPOAligner
    aligner = DPOAligner()
    aligner.train(settings.output_dir, "data/dpo_pairs.json")

    # Step 4: 评测对比
    logger.info("=" * 60)
    logger.info("Step 4: Evaluation comparison")
    logger.info("=" * 60)
    from src.evaluation.model_evaluator import ModelEvaluator
    evaluator = ModelEvaluator()
    evaluator.load_dataset()

    # 对比基座 vs SFT vs DPO
    logger.info("Evaluating base model...")
    # base_result = await evaluator.evaluate_model("base", base_generate_fn)

    logger.info("Evaluating SFT model...")
    # sft_result = await evaluator.evaluate_model("sft", sft_generate_fn)

    logger.info("Evaluating DPO model...")
    # dpo_result = await evaluator.evaluate_model("dpo", dpo_generate_fn)

    # comparison = evaluator.compare([base_result, sft_result, dpo_result])
    # logger.info(f"\nComparison:\n{json.dumps(comparison, indent=2)}")

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    asyncio.run(run_pipeline())
