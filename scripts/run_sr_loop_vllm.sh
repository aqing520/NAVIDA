#!/bin/bash
set -euo pipefail

EVAL_PYTHON="/data1/conda_envs/embAI_sup/navida_wzy/bin/python"
VLLM_PYTHON="/data1/conda_envs/embAI_sup/vllmwzy/bin/python"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

CONFIG_PATH="config/vln_r2r.yaml"
BASELINE_PATH="eval_log/navida_r2r_vllm_official_gpu0"
RESULT_FILE="result.txt"
OPENAI_API_BASE_URL="http://127.0.0.1:8201/v1"

PILOT_CHUNKS=12
PILOT_MAX_EPISODES=20
PILOT_GPUS=(5 5 5 5 6 6 6 6 0 0 0 0)

FULL_CHUNKS=12
FULL_GPUS=(5 5 5 5 6 6 6 6 0 0 0 0)

export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="$OPENAI_API_BASE_URL"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

mkdir -p eval_log/runtime_logs

check_vllm() {
    curl --noproxy '*' -sS -m 10 "$OPENAI_API_BASE_URL/models" >/dev/null
}

run_splits() {
    local label="$1"
    local result_path="$2"
    local prompt_style="$3"
    local temperature="$4"
    local chunks="$5"
    local max_episodes="$6"
    shift 6
    local gpus=("$@")

    local log_path="eval_log/runtime_logs/${label}_$(date +%m%d_%H%M%S).log"
    echo "Running $label -> $result_path" | tee -a "$log_path"
    echo "chunks=$chunks gpus=${gpus[*]} prompt_style=$prompt_style temperature=$temperature max_episodes=$max_episodes" | tee -a "$log_path"

    for idx in $(seq 0 $((chunks - 1))); do
        local gpu="${gpus[$idx]}"
        echo "Launching worker split=$idx gpu=$gpu" | tee -a "$log_path"
        if [[ "$max_episodes" == "none" ]]; then
            CUDA_VISIBLE_DEVICES="$gpu" "$EVAL_PYTHON" src/eval/eval_vllm.py \
                --exp-config "$CONFIG_PATH" \
                --split-num "$chunks" \
                --split-id "$idx" \
                --forward-distance 25 \
                --turn-angle 15 \
                --resolution-ratio 0.5 \
                --max-action-history 200 \
                --num-generations 1 \
                --prompt-style "$prompt_style" \
                --temperature "$temperature" \
                --result-path "$result_path" >>"$log_path" 2>&1 &
        else
            CUDA_VISIBLE_DEVICES="$gpu" "$EVAL_PYTHON" src/eval/eval_vllm.py \
                --exp-config "$CONFIG_PATH" \
                --split-num "$chunks" \
                --split-id "$idx" \
                --forward-distance 25 \
                --turn-angle 15 \
                --resolution-ratio 0.5 \
                --max-action-history 200 \
                --num-generations 1 \
                --prompt-style "$prompt_style" \
                --temperature "$temperature" \
                --max-episodes "$max_episodes" \
                --result-path "$result_path" >>"$log_path" 2>&1 &
        fi
    done

    wait
    echo "Finished $label" | tee -a "$log_path"
}

analyze_and_record() {
    local label="$1"
    local phase="$2"
    local result_path="$3"
    local prompt_style="$4"
    local temperature="$5"
    local max_episodes="$6"
    local worker_count="$7"
    local worker_gpus="$8"
    local summary_json="eval_log/runtime_logs/${label}_${phase}_summary.json"

    "$EVAL_PYTHON" src/eval/compare_results.py \
        --path "$result_path" \
        --baseline-path "$BASELINE_PATH" \
        --label "$label" \
        --phase "$phase" \
        --result-file "$RESULT_FILE" \
        --eval-python "$EVAL_PYTHON" \
        --vllm-python "$VLLM_PYTHON" \
        --worker-count "$worker_count" \
        --worker-gpus "$worker_gpus" \
        --prompt-style "$prompt_style" \
        --temperature "$temperature" \
        --max-episodes "$max_episodes" \
        --summary-json "$summary_json" \
        --append
}

select_best_candidate() {
    "$EVAL_PYTHON" - <<'PY'
import glob, json
best = None
for path in glob.glob("eval_log/runtime_logs/*_pilot_summary.json"):
    with open(path) as f:
        item = json.load(f)
    if item.get("decision") != "promote":
        continue
    delta_sr = item["delta"]["success_rate"]
    if best is None or delta_sr > best["delta"]["success_rate"]:
        best = item
if best is None:
    print("")
else:
    print(best["label"])
PY
}

append_no_promotion_note() {
    cat >>"$RESULT_FILE" <<'EOF'


====================================
SR提升循环实验阶段性结论

前三个pilot候选未找到满足进入全量条件的配置。
下一步决策：转向本地 eval.py 同 split 对比，排查 vLLM processor/serving 差异；不要继续扩大 prompt 改写。
=====================================
EOF
}

check_vllm

declare -a LABELS=(
    "baseline_t0p0_pilot"
    "baseline_t0p1_pilot"
    "stop_hint_t0p2_pilot"
)
declare -A PROMPTS=(
    ["baseline_t0p0_pilot"]="baseline"
    ["baseline_t0p1_pilot"]="baseline"
    ["stop_hint_t0p2_pilot"]="stop_hint"
)
declare -A TEMPS=(
    ["baseline_t0p0_pilot"]="0.0"
    ["baseline_t0p1_pilot"]="0.1"
    ["stop_hint_t0p2_pilot"]="0.2"
)
declare -A PATHS=(
    ["baseline_t0p0_pilot"]="eval_log/navida_r2r_vllm_baseline_t0p0_pilot12x20"
    ["baseline_t0p1_pilot"]="eval_log/navida_r2r_vllm_baseline_t0p1_pilot12x20"
    ["stop_hint_t0p2_pilot"]="eval_log/navida_r2r_vllm_stop_hint_t0p2_pilot12x20"
)

for label in "${LABELS[@]}"; do
    run_splits "$label" "${PATHS[$label]}" "${PROMPTS[$label]}" "${TEMPS[$label]}" \
        "$PILOT_CHUNKS" "$PILOT_MAX_EPISODES" "${PILOT_GPUS[@]}"
    analyze_and_record "$label" "pilot" "${PATHS[$label]}" "${PROMPTS[$label]}" "${TEMPS[$label]}" \
        "$PILOT_MAX_EPISODES" "$PILOT_CHUNKS" "${PILOT_GPUS[*]}"
done

best_label="$(select_best_candidate)"
if [[ -z "$best_label" ]]; then
    append_no_promotion_note
    echo "No pilot candidate met full-eval criteria."
    exit 0
fi

full_label="${best_label/_pilot/_full}"
full_path="eval_log/navida_r2r_vllm_${full_label}16"
run_splits "$full_label" "$full_path" "${PROMPTS[$best_label]}" "${TEMPS[$best_label]}" \
    "$FULL_CHUNKS" "none" "${FULL_GPUS[@]}"
analyze_and_record "$full_label" "full" "$full_path" "${PROMPTS[$best_label]}" "${TEMPS[$best_label]}" \
    "none" "$FULL_CHUNKS" "${FULL_GPUS[*]}"
