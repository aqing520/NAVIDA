#!/bin/bash

PYTHON="/data1/conda_envs/embAI_sup/awzy/navida_wzy/bin/python"
export PYTHONPATH=`pwd`:$PYTHONPATH

CONFIG_PATH="config/vln_r2r_train.yaml"
RESULT_PATH="data/rollout_train_baseline"

CHUNKS=32
gpus=(1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 4 4 4 4 4 4 4 4 5 5 5 5 5 5 5 5)

export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://127.0.0.1:8201/v1"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

echo "Starting rollout capture on train split..."
echo "Config: $CONFIG_PATH"
echo "Output: $RESULT_PATH"
echo "Workers: $CHUNKS"

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "Launching worker $IDX on GPU ${gpus[$IDX]}"
    CUDA_VISIBLE_DEVICES=${gpus[$IDX]} $PYTHON src/data/capture_rollout.py \
        --exp-config $CONFIG_PATH \
        --split-num $CHUNKS \
        --split-id $IDX \
        --forward-distance 25 \
        --turn-angle 15 \
        --resolution-ratio 0.5 \
        --max-action-history 200 \
        --temperature 0.1 \
        --result-path $RESULT_PATH &
done

wait
echo "All workers finished."
echo "Results saved to $RESULT_PATH"
