export NCCL_P2P_LEVEL=NVL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_PATH=${MODEL_PATH:-"$REPO_ROOT/result/qwen3vl4b_r2r_rxr_formal_freeze_linear_attn/serving-checkpoint-200"}
VLLM_PYTHON="/data1/conda_envs/embAI_sup/awzy/vllmwzy/bin/python"
VLLM_GPU=${VLLM_GPU:-0}
PORT=${PORT:-8201}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}

CUDA_VISIBLE_DEVICES=$VLLM_GPU $VLLM_PYTHON -m vllm.entrypoints.cli.main serve "$MODEL_PATH" --runner generate \
    --trust-remote-code --limit-mm-per-prompt '{"image": 99999}' \
    --mm-processor-kwargs '{"max_pixels": 501760}' \
    --max-model-len 32768 --max-num-batched-tokens 65536 \
    --port $PORT \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
