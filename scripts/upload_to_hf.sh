#!/bin/bash

set -e

export HF_TOKEN="your-token-here"

REPO_ID="username/sft-eval-demo"

echo "[INFO] Uploading to HuggingFace Hub..."
python -m uploader.hf_uploader \
    --repo_id $REPO_ID \
    --model_dir "outputs/my-sft-model" \
    --result_file "results/metrics.json" \
    --dataset_dir "data/eval_set"

echo "[INFO] Upload selesai!"
