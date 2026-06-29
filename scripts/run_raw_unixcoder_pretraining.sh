#!/usr/bin/env bash
set -euo pipefail

# Train the small UniXcoder-style RoBERTa baseline on raw CodeSearchNet Python
# function/docstring pairs. This intentionally does not consume JEPA triples.

PYTHON_BIN="${PYTHON_BIN:-python}"
TOKENIZER_DIR="${TOKENIZER_DIR:-assets/tokenizers/codesearchnet-python/bpe16k}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/pretrain-unixcoder-raw}"

DATASET_NAME="${DATASET_NAME:-code_search_net}"
DATASET_CONFIG="${DATASET_CONFIG:-python}"
SPLIT="${SPLIT:-train}"
LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR:-}"
DATA_FILES="${DATA_FILES:-}"
STREAMING="${STREAMING:-1}"

STEPS="${STEPS:-200000}"
DURATION_HOURS="${DURATION_HOURS:-0}"
MAX_LEN="${MAX_LEN:-256}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-2e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
MLM_PROBABILITY="${MLM_PROBABILITY:-0.15}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.1}"
TEMPERATURE="${TEMPERATURE:-0.05}"
PRECISION="${PRECISION:-bf16}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
EVAL_BATCHES="${EVAL_BATCHES:-20}"
SEED="${SEED:-123456}"
GPU="${GPU:-0}"
DRY_RUN_BATCHES="${DRY_RUN_BATCHES:-0}"

mkdir -p "$OUTPUT_DIR"

ARGS=(
  scripts/train_small_unixcoder_raw.py
  --output-dir "$OUTPUT_DIR"
  --model-name "$TOKENIZER_DIR"
  --dataset-name "$DATASET_NAME"
  --dataset-config "$DATASET_CONFIG"
  --split "$SPLIT"
  --max-len "$MAX_LEN"
  --batch-size "$BATCH_SIZE"
  --steps "$STEPS"
  --duration-hours "$DURATION_HOURS"
  --lr "$LR"
  --warmup-steps "$WARMUP_STEPS"
  --mlm-probability "$MLM_PROBABILITY"
  --contrastive-weight "$CONTRASTIVE_WEIGHT"
  --temperature "$TEMPERATURE"
  --precision "$PRECISION"
  --eval-every "$EVAL_EVERY"
  --eval-batches "$EVAL_BATCHES"
  --save-every "$SAVE_EVERY"
  --seed "$SEED"
)

if [[ "$STREAMING" != "1" ]]; then
  ARGS+=(--no-streaming)
fi

if [[ -n "$LOCAL_DATASET_DIR" ]]; then
  ARGS+=(--local-dataset-dir "$LOCAL_DATASET_DIR")
fi

if [[ -n "$DATA_FILES" ]]; then
  # shellcheck disable=SC2206
  files=($DATA_FILES)
  ARGS+=(--data-files "${files[@]}")
fi

if [[ "$DRY_RUN_BATCHES" != "0" ]]; then
  ARGS+=(--dry-run-batches "$DRY_RUN_BATCHES")
fi

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" "${ARGS[@]}"

echo "Done. Checkpoint:"
echo "  $OUTPUT_DIR/latest.pt"
