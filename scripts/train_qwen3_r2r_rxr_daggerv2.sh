export PYTHONPATH="./:$PYTHONPATH"
export NCCL_P2P_LEVEL=NVL
export NAVIDA_DEBUG_RAW_LOSS="${NAVIDA_DEBUG_RAW_LOSS:-0}"
export NAVIDA_DEBUG_RAW_LOSS_STEPS="${NAVIDA_DEBUG_RAW_LOSS_STEPS:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/models/qwen3vl4b_ckpt12765}"
TRAIN_FILE="${TRAIN_FILE:-$REPO_ROOT/data/train_r2r_rxr_qwen3vl4b_full_with_daggerv2.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/result/qwen3vl4b_r2r_rxr_daggerv2_zero2}"
LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/train.log}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-/data1/conda_envs/embAI_sup/awzy/navida_wzy/bin/deepspeed}"
MASTER_PORT="${MASTER_PORT:-25436}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-1,2,3,4}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
ENABLE_DATALOADER_PIN_MEMORY="${ENABLE_DATALOADER_PIN_MEMORY:-0}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
LEARNING_RATE="${LEARNING_RATE:-5.0e-6}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

cmd=(
    "$DEEPSPEED_BIN" --master_port "$MASTER_PORT" src/train/train.py
    --deepspeed scripts/zero2.json
    --dataset_name "$TRAIN_FILE"
    --model_name_or_path "$MODEL_PATH"
    --num_train_epochs "$NUM_TRAIN_EPOCHS"
    --bf16
    --torch_dtype bfloat16
    --attn_implementation flash_attention_2
    --lr_scheduler_type cosine
    --gradient_checkpointing True
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
    --learning_rate "$LEARNING_RATE"
    --logging_steps 5
    --eval_strategy no
    --save_strategy steps
    --save_steps "$SAVE_STEPS"
    --save_total_limit "$SAVE_TOTAL_LIMIT"
    --output_dir "$OUTPUT_DIR"
    --report_to tensorboard
)

if [[ "$ENABLE_DATALOADER_PIN_MEMORY" == "1" ]]; then
    cmd+=(--dataloader_pin_memory)
fi

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
    cmd+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" "${cmd[@]}"
