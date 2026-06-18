#!/bin/bash

PYTHON="/data1/conda_envs/embAI_sup/awzy/navida_wzy/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

SIGLIP_MODEL_PATH=${SIGLIP_MODEL_PATH:-"/data1/dataset/embAI_sup/siglip-so400m-patch14-384"}
SIGLIP_GPU=${SIGLIP_GPU:-0}
SIGLIP_HOST=${SIGLIP_HOST:-"127.0.0.1"}
SIGLIP_PORT=${SIGLIP_PORT:-8301}
SIGLIP_DEVICE=${SIGLIP_DEVICE:-"cuda"}

CUDA_VISIBLE_DEVICES=$SIGLIP_GPU $PYTHON src/eval/siglip_embedding_server.py \
    --model-path "$SIGLIP_MODEL_PATH" \
    --device "$SIGLIP_DEVICE" \
    --host "$SIGLIP_HOST" \
    --port "$SIGLIP_PORT"
