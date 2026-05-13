export PYTHONPATH="./:$PYTHONPATH"
export NCCL_P2P_LEVEL=NVL
export NAVIDA_DEBUG_RAW_LOSS="${NAVIDA_DEBUG_RAW_LOSS:-0}"
export NAVIDA_DEBUG_RAW_LOSS_STEPS="${NAVIDA_DEBUG_RAW_LOSS_STEPS:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/models/qwen2.5vl}"
TRAIN_FILE="${TRAIN_FILE:-$REPO_ROOT/data/train_r2r_rxr_qwen3vl4b_full.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/result/qwen25vl3b_r2r_streamvln_rxr}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-/data1/conda_envs/embAI_sup/awzy/navida_wzy/bin/deepspeed}"
MASTER_PORT="${MASTER_PORT:-25437}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0,1,2,3}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
SAVE_STEPS="${SAVE_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"

cd "$REPO_ROOT"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" "$DEEPSPEED_BIN" --master_port "$MASTER_PORT" src/train/train.py \
    --deepspeed scripts/zero2.json \
    --dataset_name "$TRAIN_FILE" \
    --model_name_or_path "$MODEL_PATH" \
    --num_train_epochs 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing True \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory \
    --learning_rate 2.0e-5 \
    --logging_steps 5 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --output_dir "$OUTPUT_DIR" \
    --report_to tensorboard
