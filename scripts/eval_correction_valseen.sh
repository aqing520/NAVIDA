#!/bin/bash

PYTHON="/data1/conda_envs/embAI_sup/awzy/navida_wzy/bin/python"
export PYTHONPATH=`pwd`:$PYTHONPATH

CONFIG_PATH="config/vln_r2r.yaml"
PROMPT_STYLE="baseline"
SAVE_PATH="eval_log/correction_flywheel_v3_step1000_valseen"

CHUNKS=32
gpus=(0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3)

export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://127.0.0.1:8201/v1"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

echo "Starting eval on val_seen..."
echo "Config: $CONFIG_PATH"
echo "Output: $SAVE_PATH"
echo "Workers: $CHUNKS on GPU 0,1,2,3"

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "Launching worker $IDX on GPU ${gpus[$IDX]}"
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
    --result-path $SAVE_PATH &
done

wait

$PYTHON src/eval/analyze_results.py \
    --path $SAVE_PATH
