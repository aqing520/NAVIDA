#!/bin/bash

MODEL_PATH="${MODEL_PATH:-models/navida_qwen2_5_vl}"
LORA_PATH="${LORA_PATH:-result/navida_correctnav_v8_lora/checkpoint-228}"
PYTHON="${PYTHON:-/data1/conda_envs/embAI_sup/awzy/navida_wzy/bin/python}"
export PYTHONPATH="$(pwd):$PYTHONPATH"

CONFIG_PATH="${CONFIG_PATH:-config/vln_r2r_val_unseen.yaml}"
PROMPT_STYLE="${PROMPT_STYLE:-baseline}"
SAVE_PATH="${SAVE_PATH:-eval_log/navida_correctnav_v8_ckpt228_val_unseen_full8}"

CHUNKS=8
gpus=(0 1 2 3 4 5 6 7)

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "Launching split $IDX on GPU ${gpus[$IDX]}"
    CUDA_VISIBLE_DEVICES=${gpus[$IDX]} $PYTHON src/eval/eval.py \
        --exp-config $CONFIG_PATH \
        --split-num $CHUNKS \
        --split-id $IDX \
        --forward-distance 25 \
        --turn-angle 15 \
        --resolution-ratio 0.5 \
        --max-action-history 200 \
        --model-path $MODEL_PATH \
        --lora-path $LORA_PATH \
        --num-generations 1 \
        --prompt-style $PROMPT_STYLE \
        --result-path $SAVE_PATH &
done

wait

$PYTHON src/eval/analyze_results.py \
    --path $SAVE_PATH
