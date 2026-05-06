#!/bin/bash

SAVE_PATH=eval_log/navida_correctnav_v8_ckpt228_val_unseen_full8

echo $SAVE_PATH
python src/eval/analyze_results.py \
    --path $SAVE_PATH
