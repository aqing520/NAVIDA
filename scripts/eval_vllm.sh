#!/bin/bash

PYTHON="/data1/conda_envs/embAI_sup/awzy/navida_wzy/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

#R2R
CONFIG_PATH=${CONFIG_PATH:-"config/vln_r2r_seen.yaml"}
PROMPT_STYLE=${PROMPT_STYLE:-"baseline"} # baseline, stop_hint, or sr_stop
MEMORY_STYLE=${MEMORY_STYLE:-"none"} # none, topo_text, or topo_semantic_text
MEMORY_ENCODER_PATH=${MEMORY_ENCODER_PATH:-"/data1/dataset/embAI_sup/siglip-so400m-patch14-384"}
MEMORY_ENCODER_DEVICE=${MEMORY_ENCODER_DEVICE:-"cpu"}
MEMORY_ENCODER_SERVER=${MEMORY_ENCODER_SERVER:-""}
MEMORY_SIM_THRESHOLD=${MEMORY_SIM_THRESHOLD:-"0.84"}
MEMORY_MAX_NODES=${MEMORY_MAX_NODES:-"80"}
MEMORY_TEXT_MAX_LINES=${MEMORY_TEXT_MAX_LINES:-"4"}
HISTORY_SELECTION=${HISTORY_SELECTION:-"uniform"}
STOP_VERIFIER=${STOP_VERIFIER:-"none"}
STOP_VERIFIER_BASE_URL=${STOP_VERIFIER_BASE_URL:-""}
TARGET_STOP_HINT=${TARGET_STOP_HINT:-"none"}
TARGET_HINT_MIN_STEP=${TARGET_HINT_MIN_STEP:-"8"}
TARGET_HINT_TOP_K=${TARGET_HINT_TOP_K:-"3"}
TARGET_HINT_MARGIN=${TARGET_HINT_MARGIN:-"0.01"}
if [ "$MEMORY_STYLE" = "none" ]; then
    SAVE_PATH=${SAVE_PATH:-"eval_log/qwen3vl4b_ckpt200_r2r_valseen_vllm_gpu0_${PROMPT_STYLE}"}
else
    SAVE_PATH=${SAVE_PATH:-"eval_log/qwen3vl4b_ckpt200_r2r_valseen_vllm_gpu0_${PROMPT_STYLE}_${MEMORY_STYLE}"}
fi
RUN_LOG="${SAVE_PATH}/eval_runner.log"
if [ -n "$MEMORY_ENCODER_SERVER" ]; then
    MEMORY_ENCODER_SERVER_ARG=(--memory-encoder-server "$MEMORY_ENCODER_SERVER")
else
    MEMORY_ENCODER_SERVER_ARG=()
fi
if [ -n "$STOP_VERIFIER_BASE_URL" ]; then
    STOP_VERIFIER_BASE_URL_ARG=(--stop-verifier-base-url "$STOP_VERIFIER_BASE_URL")
else
    STOP_VERIFIER_BASE_URL_ARG=()
fi

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
    --memory-style $MEMORY_STYLE \
    --memory-encoder-path $MEMORY_ENCODER_PATH \
    --memory-encoder-device $MEMORY_ENCODER_DEVICE \
    "${MEMORY_ENCODER_SERVER_ARG[@]}" \
    --memory-sim-threshold $MEMORY_SIM_THRESHOLD \
    --memory-max-nodes $MEMORY_MAX_NODES \
    --memory-text-max-lines $MEMORY_TEXT_MAX_LINES \
    --history-selection $HISTORY_SELECTION \
    --stop-verifier $STOP_VERIFIER \
    "${STOP_VERIFIER_BASE_URL_ARG[@]}" \
    --target-stop-hint $TARGET_STOP_HINT \
    --target-hint-min-step $TARGET_HINT_MIN_STEP \
    --target-hint-top-k $TARGET_HINT_TOP_K \
    --target-hint-margin $TARGET_HINT_MARGIN \
    --result-path $SAVE_PATH >> "$RUN_LOG" 2>&1 &
done

wait

$PYTHON src/eval/analyze_results.py \
    --path $SAVE_PATH >> "$RUN_LOG" 2>&1
