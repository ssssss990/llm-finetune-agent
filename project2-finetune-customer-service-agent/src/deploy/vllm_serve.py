"""
vLLM 部署脚本

vLLM + PagedAttention 优化推理吞吐
关注指标：QPS（每秒请求数）、TTFT（首 token 延迟）、端到端延迟

启动方式：
python -m vllm.entrypoints.openai.api_server \
    --model outputs/dpo \
    --port 8002 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096
"""

import subprocess
import sys
from config.settings import settings


def start_vllm_server(
    model_path: str = None,
    port: int = None,
):
    """启动 vLLM 推理服务"""
    model_path = model_path or settings.dpo_output_dir
    port = port or settings.vllm_port

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--gpu-memory-utilization", str(settings.vllm_gpu_memory_utilization),
        "--max-model-len", str(settings.vllm_max_model_len),
        "--dtype", "bfloat16",
        "--enable-prefix-caching",
    ]

    print(f"Starting vLLM server: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    start_vllm_server(args.model, args.port)
