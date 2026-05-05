#!/bin/bash

PYTHON="/data1/conda_envs/embAI_sup/awzy/navida_wzy/bin/python"
export PYTHONPATH=`pwd`:$PYTHONPATH

# Configuration
ROLLOUT_DIR="data/rollout_train_baseline"
ERRORS_PATH="${ROLLOUT_DIR}/errors_type1.jsonl"
CORRECTION_PATH="${ROLLOUT_DIR}/correction_train_type1.jsonl"
MIXED_PATH="data/mixed_vln_correction_train.jsonl"
ORIGINAL_DATA="data/only_r2r_idm_vln_mix_data.jsonl"

# Check if original training data exists
if [ ! -f "$ORIGINAL_DATA" ]; then
    echo "ERROR: Original training data not found: $ORIGINAL_DATA"
    echo "Please run prepare_training_data.py first or update the path."
    exit 1
fi

# Exp A: stop correction only (Type 1)
echo "=== Stage 2: Mining errors (Type 1 only) ==="
$PYTHON src/data/mine_errors.py \
    --rollout-dir $ROLLOUT_DIR \
    --output $ERRORS_PATH \
    --goal-radius 3.0 \
    --error-types type1

echo ""
echo "=== Stage 3: Building correction data ==="
$PYTHON src/data/build_correction_data.py \
    --errors $ERRORS_PATH \
    --rollout-dir $ROLLOUT_DIR \
    --output $CORRECTION_PATH

echo ""
echo "=== Stage 4: Mixing datasets ==="
$PYTHON src/data/mix_dataset.py \
    --original $ORIGINAL_DATA \
    --correction $CORRECTION_PATH \
    --output $MIXED_PATH \
    --original-ratio 0.9 \
    --correction-ratio 0.1

echo ""
echo "=== Done ==="
echo "Mixed dataset: $MIXED_PATH"
echo "Use this for training with train_correction.sh"
