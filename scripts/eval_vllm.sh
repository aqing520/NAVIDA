#!/bin/bash

PYTHON="/data1/conda_envs/embAI_sup/navida_wzy/bin/python"
export PYTHONPATH=`pwd`:$PYTHONPATH

#R2R
CONFIG_PATH="config/vln_r2r.yaml"
SAVE_PATH="eval_log/navida_r2r_vllm_official_gpu0"

#RxR
# CONFIG_PATH="config/vln_rxr.yaml"
# SAVE_PATH="eval_log/navida_rxr" 

CHUNKS=4 # number of Habitat workers / dataset splits
gpus=(0 0 0 0)

export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://127.0.0.1:8201/v1"
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
    --result-path $SAVE_PATH &
done

wait

$PYTHON src/eval/analyze_results.py \
    --path $SAVE_PATH
