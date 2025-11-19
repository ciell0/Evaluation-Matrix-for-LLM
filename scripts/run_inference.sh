#!/bin/bash

set -e

MODEL_PATH="outputs/my-sft-model"
DATA_PATH="data/eval_set.jsonl"
OUTPUT_PATH="results/predictions.jsonl"

echo "[INFO] Running inference..."
python -m src.inference.run_inference \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_path $OUTPUT_PATH

echo "[INFO] Inference selesai. Output disimpan di $OUTPUT_PATH"
