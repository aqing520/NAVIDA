#!/bin/bash

PYTHON="/data1/conda_envs/embAI_sup/awzy/navida_wzy/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

#R2R
CONFIG_PATH=${CONFIG_PATH:-"config/vln_r2r_seen.yaml"}
PROMPT_STYLE=${PROMPT_STYLE:-"baseline"} # baseline, stop_hint, or sr_stop
SAVE_PATH=${SAVE_PATH:-"eval_log/qwen3vl4b_ckpt200_r2r_valseen_vllm_gpu0_${PROMPT_STYLE}"}
RUN_LOG="${SAVE_PATH}/eval_runner.log"

#RxR
# CONFIG_PATH="config/vln_rxr.yaml"
# SAVE_PATH="eval_log/navida_rxr" 

CHUNKS=${CHUNKS:-2} # conservative default because GPU 7 is partially occupied on this machine
HABITAT_GPU=${HABITAT_GPU:-7}

export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://127.0.0.1:8201/v1"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

mkdir -p "$SAVE_PATH"
echo "Evaluation log: $RUN_LOG" | tee -a "$RUN_LOG"

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "Launching Habitat worker $IDX on GPU ${HABITAT_GPU}" | tee -a "$RUN_LOG"
    CUDA_VISIBLE_DEVICES=${HABITAT_GPU} $PYTHON src/eval/eval_vllm.py \
    --exp-config $CONFIG_PATH \
    --split-num $CHUNKS \
    --split-id $IDX \
    --forward-distance 25 \
    --turn-angle 15 \
    --resolution-ratio 0.5 \
    --max-action-history 200 \
    --num-generations 1 \
    --prompt-style $PROMPT_STYLE \
    --result-path $SAVE_PATH >> "$RUN_LOG" 2>&1 &
done

wait

$PYTHON src/eval/analyze_results.py \
    --path $SAVE_PATH >> "$RUN_LOG" 2>&1
