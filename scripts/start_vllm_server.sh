export NCCL_P2P_LEVEL=NVL

MODEL_PATH="models/navida_qwen2_5_vl"
VLLM="/data1/conda_envs/embAI_sup/vllmwzy/bin/vllm"
VLLM_GPU=${VLLM_GPU:-0}
PORT=${PORT:-8201}

CUDA_VISIBLE_DEVICES=$VLLM_GPU $VLLM serve "$MODEL_PATH" --runner generate \
    --trust-remote-code --limit-mm-per-prompt '{"image": 99999}' \
    --mm-processor-kwargs '{"max_pixels": 501760}' \
    --max-model-len 32768 --max-num-batched-tokens 65536 \
    --port $PORT \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
