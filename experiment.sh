#!/bin/bash

set -e

MODEL_PATH="outputs/my-sft-model"
DATASET="hf://username/my-eval-dataset"
PRED_PATH="results/predictions.jsonl"
METRICS_PATH="results/metrics.json"
HF_REPO="username/sft-eval-demo"

echo "========================================"
echo "[1] Running inference"
echo "========================================"
python -m src.inference.run_inference \
    --model_path $MODEL_PATH \
    --data_path $DATASET \
    --output_path $PRED_PATH

echo "========================================"
echo "[2] Running evaluation metrics"
echo "========================================"
python -m src.eval.evaluator \
    --pred_path $PRED_PATH \
    --gold_path $DATASET \
    --config configs/eval_config.yaml \
    --output_path $METRICS_PATH

echo "========================================"
echo "[3] Uploading results to HuggingFace Hub"
echo "========================================"
python -m uploader.hf_uploader \
    --repo_id $HF_REPO \
    --model_dir $MODEL_PATH \
    --result_file $METRICS_PATH

echo "========================================"
echo "Pipeline selesai 🚀"
