#!/bin/bash

export PYTHONPATH="./:$PYTHONPATH"
export PATH="/data1/conda_envs/embAI_sup/awzy/navida_wzy/bin:$PATH"

ROLLOUT_DIR="data/rollout_train_baseline"
ANNOTATIONS_PATH="data/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz"
EXP_CONFIG="config/vln_r2r_train.yaml"

ANCHOR_DATA="data/correction_nav_v8/anchor_train.jsonl"
CORRECTION_DATA="data/correction_nav_v8/correction_train.jsonl"
MIXED_DATA="data/correction_nav_v8/mixed_anchor_correction.jsonl"

mkdir -p data/correction_nav_v8

python src/data/build_anchor_dataset_v8.py \
    --rollout-dir "$ROLLOUT_DIR" \
    --annotations-path "$ANNOTATIONS_PATH" \
    --output "$ANCHOR_DATA" \
    --threshold 1.5 \
    --max-history 8 \
    --max-success-samples 120000 \
    --max-prefix-samples 20000

python src/data/build_correction_nav_v8.py \
    --rollout-dir "$ROLLOUT_DIR" \
    --annotations-path "$ANNOTATIONS_PATH" \
    --exp-config "$EXP_CONFIG" \
    --output "$CORRECTION_DATA" \
    --threshold 1.5 \
    --max-history 8 \
    --max-recovery-steps 6 \
    --min-deviation-distance 1.5 \
    --max-deviation-distance 3.5 \
    --min-goal-distance 2.0 \
    --max-goal-distance 20.0 \
    --min-progress 0.5 \
    --goal-radius 3.0

python src/data/mix_dataset.py \
    --original "$ANCHOR_DATA" \
    --correction "$CORRECTION_DATA" \
    --output "$MIXED_DATA" \
    --original-ratio 0.8 \
    --correction-ratio 0.2
