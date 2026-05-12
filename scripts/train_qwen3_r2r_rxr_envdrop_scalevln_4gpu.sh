export NCCL_P2P_LEVEL=NVL
export NAVIDA_DEBUG_RAW_LOSS=0
export NAVIDA_DEBUG_RAW_LOSS_STEPS=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_PATH="$REPO_ROOT/models/Qwen3-VL-4B-Instruct"
TRAIN_FILE="$REPO_ROOT/data/train_r2r_rxr_envdrop_scalevln_qwen3vl4b_full.jsonl"
OUTPUT_DIR="$REPO_ROOT/result/qwen3vl4b_r2r_rxr_envdrop_scalevln_4gpu"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

if [ ! -f "$TRAIN_FILE" ]; then
  echo "Training file not found: $TRAIN_FILE"
  exit 1
fi

CUDA_VISIBLE_DEVICES=3,4,5,6 /data1/conda_envs/embAI_sup/awzy/navida_wzy/bin/python -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port 25436 \
    "$REPO_ROOT/src/train/train.py" \
    --dataset_name "$TRAIN_FILE" \
    --model_name_or_path "$MODEL_PATH" \
    --num_train_epochs 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --attn_implementation sdpa \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.01 \
    --gradient_checkpointing True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --dataloader_num_workers 0 \
    --dataloader_pin_memory False \
    --learning_rate 5.0e-6 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0 \
    --logging_steps 10 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 5 \
    --output_dir "$OUTPUT_DIR" \
    --report_to tensorboard
