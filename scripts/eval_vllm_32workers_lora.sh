#!/bin/bash

PYTHON="${PYTHON:-/data1/conda_envs/embAI_sup/awzy/navida_wzy/bin/python}"
export PYTHONPATH="$(pwd):$PYTHONPATH"

CONFIG_PATH="${CONFIG_PATH:-config/vln_r2r.yaml}"
PROMPT_STYLE="${PROMPT_STYLE:-baseline}"
SAVE_PATH="${SAVE_PATH:-eval_log/navida_correctnav_v8_ckpt228_vllm_valseen_full32}"
CHUNKS="${CHUNKS:-32}"
MODEL_ID="${MODEL_ID:-correctnav_v8_ckpt228}"

gpus=(0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3)

export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8201/v1}"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "Launching Habitat worker $IDX on GPU ${gpus[$IDX]}"
    CUDA_VISIBLE_DEVICES=${gpus[$IDX]} $PYTHON src/eval/eval_vllm.py \
        --exp-config $CONFIG_PATH \
        --split-num $CHUNKS \
        --split-id $IDX \
        --forward-distance 25 \
        --turn-angle 15 \
        --resolution-ratio 0.5 \
        --max-action-history 200 \
        --num-generations 1 \
        --prompt-style $PROMPT_STYLE \
        --model-id $MODEL_ID \
        --result-path $SAVE_PATH &
done

wait

$PYTHON src/eval/analyze_results.py --path $SAVE_PATH
