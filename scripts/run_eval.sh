#!/bin/bash

set -e

PRED_PATH="results/predictions.jsonl"
GOLD_PATH="data/eval_set.jsonl"
CONFIG="configs/eval_config.yaml"
SAVE_RESULT="results/metrics.json"

echo "[INFO] Running evaluation..."
python -m src.eval.evaluator \
    --pred_path $PRED_PATH \
    --gold_path $GOLD_PATH \
    --config $CONFIG \
    --output_path $SAVE_RESULT

echo "[INFO] Evaluasi selesai. Metrik disimpan di $SAVE_RESULT"
