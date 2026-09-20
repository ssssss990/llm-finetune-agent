"""
微调与部署配置
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: str = "outputs/sft"
    dpo_output_dir: str = "outputs/dpo"

    # LoRA 参数
    lora_r: int = 64
    lora_alpha: int = 128
    lora_target_modules: list[str] = ["q_proj", "v_proj", "down_proj"]
    lora_dropout: float = 0.05

    # 量化
    use_qlora: bool = True
    quantization_bits: int = 4

    # 训练参数
    num_epochs: int = 3
    learning_rate: float = 2e-4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    batch_size: int = 2

    # DPO 参数
    dpo_beta: float = 0.1
    dpo_epochs: int = 1
    dpo_learning_rate: float = 5e-6
    dpo_batch_size: int = 1

    # 数据
    train_data_path: str = "data/sft_train.json"
    eval_data_path: str = "data/sft_eval.json"
    dpo_data_path: str = "data/dpo_pairs.json"

    # 部署
    vllm_port: int = 8002
    vllm_gpu_memory_utilization: float = 0.85
    vllm_max_model_len: int = 4096

    # 评测
    eval_dataset_path: str = "data/eval/eval_300.json"

    class Config:
        env_file = ".env"
        env_prefix = "FT_"


settings = Settings()
