"""
数据生成脚本

用法：
python scripts/generate_data.py

生成：
- data/sft_train.json (3000 条 SFT 样本)
- data/sft_eval.json (300 条评估样本)
- data/dpo_pairs.json (800 条偏好对)
"""

import json
from src.data.data_generator import CustomerServiceDataGenerator


def main():
    gen = CustomerServiceDataGenerator()
    samples = gen.generate(num_samples=3000)

    # 分割训练集和评估集
    train = samples[:2700]
    eval_samples = samples[2700:]

    gen.save_to_sft_format(train, "data/sft_train.json")
    gen.save_to_sft_format(eval_samples, "data/sft_eval.json")

    # 生成 DPO 偏好对
    dpo_pairs = gen.generate_dpo_pairs(samples, num_pairs=800)
    gen.save_dpo_pairs(dpo_pairs, "data/dpo_pairs.json")

    # 生成评测集
    eval_data = []
    for s in eval_samples[:300]:
        eval_data.append({
            "query": s.query,
            "intent": s.intent,
            "expected_answer": s.response,
            "is_adversarial": s.is_adversarial,
        })
    with open("data/eval/eval_300.json", "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(eval_data)} eval samples to data/eval/eval_300.json")


if __name__ == "__main__":
    main()
