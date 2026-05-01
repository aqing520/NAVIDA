#!/bin/bash
set -euo pipefail

PYTHON="/data1/conda_envs/embAI_sup/awzy/navida_wzy/bin/python"
ROOT="/data1/code/embAI_sup/awzy/NAVIDA"
TS="$(date +%m%d_%H%M%S)"
SAVE_PATH="eval_log/quct_turncap2_t0p1_24w_${TS}"

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8201/v1}"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

mkdir -p eval_log/runtime_logs "$SAVE_PATH"

gpus=(1 1 1 1 1 1 4 4 4 4 4 4 5 5 5 5 5 5 6 6 6 6 6 6)

for IDX in $(seq 0 23); do
    echo "Launching worker $IDX on GPU ${gpus[$IDX]}"
    CUDA_VISIBLE_DEVICES=${gpus[$IDX]} "$PYTHON" src/eval/eval_vllm.py \
        --exp-config config/vln_r2r.yaml \
        --split-num 24 \
        --split-id "$IDX" \
        --scene-id QUCTc6BB5sX \
        --forward-distance 25 \
        --turn-angle 15 \
        --resolution-ratio 0.5 \
        --max-action-history 200 \
        --num-generations 1 \
        --prompt-style baseline \
        --temperature 0.1 \
        --result-path "$SAVE_PATH" \
        > "eval_log/runtime_logs/turncap24_split_${IDX}_${TS}.log" 2>&1 &
done

wait

"$PYTHON" src/eval/analyze_results.py --path "$SAVE_PATH" \
    > "eval_log/runtime_logs/turncap24_analyze_${TS}.log" 2>&1

echo "$SAVE_PATH"
