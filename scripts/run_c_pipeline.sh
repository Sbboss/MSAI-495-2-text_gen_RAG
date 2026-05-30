#!/usr/bin/env bash
set -euo pipefail

# Run C: high-quality filtering + grounded_qa prompt template, then training.
# Usage:
#   bash scripts/run_c_pipeline.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "[Run C] Cleaning previous tokenized outputs..."
rm -rf data/processed/tokenizer data/processed/tokenized
rm -f data/processed/qa_raw_train.parquet data/processed/qa_raw_val.parquet data/processed/qa_raw_test.parquet

echo "[Run C] Preprocessing with quality filters + grounded_qa template..."
python scripts/preprocess_stackoverflow_fast.py \
  --processed-dir data/processed \
  --memory-limit 32GB \
  --threads 8 \
  --workers 1 \
  --encode-batch-size 8192 \
  --rows-per-chunk-file 65536 \
  --vocab-size 32000 \
  --tokenizer-sample-rows 2000000 \
  --val-fraction 0.10 \
  --test-fraction 0.05 \
  --min-answer-score 1 \
  --min-answer-chars 80 \
  --min-answer-words 12 \
  --drop-short-thanks \
  --prompt-style grounded_qa \
  --cleanup-parts

echo "[Run C] Sanity check..."
python scripts/check_preprocess_layout.py --project-root .

echo "[Run C] Starting training..."
python scripts/train_gpt2_jax.py --params params.yml
