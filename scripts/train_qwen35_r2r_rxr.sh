export PYTHONPATH="./:$PYTHONPATH"
export NCCL_P2P_LEVEL=NVL
export NAVIDA_DEBUG_RAW_LOSS=1
export NAVIDA_DEBUG_RAW_LOSS_STEPS=200

TRAIN_FILE=data/train_r2r_rxr_qwen35_full.jsonl

if [ ! -f "$TRAIN_FILE" ]; then
  cat data/navida_train_data_r2r.jsonl data/navida_train_data_streamvln_rxr.jsonl > "$TRAIN_FILE"
fi

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 /data1/conda_envs/embAI_sup/awzy/navida_qwen35/bin/python -m deepspeed.launcher.runner \
    --master_port 25435 \
    src/train/train.py \
    --deepspeed scripts/zero2.json \
    --dataset_name "$TRAIN_FILE" \
    --model_name_or_path /data1/code/embAI_sup/awzy/models/qwen3.5-4b \
    --num_train_epochs 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.01 \
    --gradient_checkpointing True \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 4 \
    --dataloader_num_workers 0 \
    --dataloader_pin_memory False \
    --learning_rate 1.0e-5 \
    --weight_decay 0.0 \
    --logging_steps 1 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 5 \
    --output_dir result/qwen35_r2r_rxr_formal_freeze_linear_attn \
    --report_to tensorboard
