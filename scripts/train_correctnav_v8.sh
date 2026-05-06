#!/bin/bash

export PYTHONPATH="./:$PYTHONPATH"
export NCCL_P2P_LEVEL=NVL
export PATH="/data1/conda_envs/embAI_sup/awzy/navida_wzy/bin:$PATH"

MIXED_DATA="data/correction_nav_v8/mixed_anchor_correction.jsonl"
BASE_MODEL="models/navida_qwen2_5_vl"
OUTPUT_DIR="result/navida_correctnav_v8_lora"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 deepspeed --master_port 25420 src/train/train.py \
    --deepspeed scripts/zero2.json \
    --dataset_name $MIXED_DATA \
    --model_name_or_path $BASE_MODEL \
    --num_train_epochs 2 \
    --bf16 \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --gradient_checkpointing True \
    --lr_scheduler_type cosine \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory \
    --learning_rate 1.0e-5 \
    --logging_steps 5 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 100 \
    --output_dir $OUTPUT_DIR \
    --report_to tensorboard \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05
