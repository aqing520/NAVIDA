#!/bin/bash

export PYTHONPATH="./:$PYTHONPATH"
export NCCL_P2P_LEVEL=NVL
export PATH="/data1/conda_envs/embAI_sup/awzy/navida_wzy/bin:$PATH"

MIXED_DATA="data/correction_expert/mixed_expert_correction.jsonl"
BASE_MODEL="models/navida_qwen2_5_vl"
OUTPUT_DIR="result/navida_correction_v4"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 deepspeed --master_port 25420 src/train/train.py \
    --deepspeed scripts/zero2.json \
    --dataset_name $MIXED_DATA \
    --model_name_or_path $BASE_MODEL \
    --resume_from_checkpoint result/navida_correction_v4/checkpoint-1000 \
    --num_train_epochs 2 \
    --bf16 \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing True \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory \
    --learning_rate 5.0e-6 \
    --logging_steps 5 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --output_dir $OUTPUT_DIR \
    --report_to tensorboard
