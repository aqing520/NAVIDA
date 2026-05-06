#!/bin/bash

export NCCL_P2P_LEVEL=NVL

MODEL_PATH="${MODEL_PATH:-models/navida_qwen2_5_vl}"
LORA_NAME="${LORA_NAME:-correctnav_v8_ckpt228}"
LORA_PATH="${LORA_PATH:-result/navida_correctnav_v8_lora/checkpoint-228}"
VLLM_PYTHON="${VLLM_PYTHON:-/data1/conda_envs/embAI_sup/awzy/vllmwzy/bin/python}"
VLLM_GPU="${VLLM_GPU:-4,5}"
TP_SIZE="${TP_SIZE:-2}"
PORT="${PORT:-8201}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

CUDA_VISIBLE_DEVICES=$VLLM_GPU $VLLM_PYTHON -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
    --runner generate \
    --trust-remote-code \
    --limit-mm-per-prompt '{"image": 99999}' \
    --mm-processor-kwargs '{"max_pixels": 501760}' \
    --max-model-len 32768 \
    --max-num-batched-tokens 65536 \
    --port $PORT \
    --tensor-parallel-size $TP_SIZE \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --enable-lora \
    --max-lora-rank 16 \
    --lora-modules ${LORA_NAME}=${LORA_PATH}
