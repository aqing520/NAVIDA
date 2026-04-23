#!/bin/bash

# sleep 7h

MODEL_PATH="models/navida_qwen2_5_vl"
PYTHON="/data1/conda_envs/embAI_sup/navida_wzy/bin/python"
export PYTHONPATH=`pwd`:$PYTHONPATH

#R2R
CONFIG_PATH="config/vln_r2r.yaml"
PROMPT_STYLE="baseline" # baseline, stop_hint, or sr_stop
SAVE_PATH="eval_log/navida_r2r_${PROMPT_STYLE}"


#RxR
# CONFIG_PATH="config/vln_rxr.yaml"
# SAVE_PATH="eval_log/navida_rxr"

# gpus available
CHUNKS=8
gpus=(2 2 3 3 4 4 5 5)

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo ${gpus[$IDX]}
    CUDA_VISIBLE_DEVICES=${gpus[$IDX]} $PYTHON src/eval/eval.py \
    --exp-config $CONFIG_PATH \
    --split-num $CHUNKS \
    --split-id $IDX \
    --forward-distance 25 \
    --turn-angle 15 \
    --resolution-ratio 0.5 \
    --max-action-history 200 \
    --model-path $MODEL_PATH \
    --num-generations 1 \
    --prompt-style $PROMPT_STYLE \
    --result-path $SAVE_PATH &
    
done

wait

$PYTHON src/eval/analyze_results.py \
    --path $SAVE_PATH
