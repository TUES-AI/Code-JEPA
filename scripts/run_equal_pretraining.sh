#!/usr/bin/env bash
set -euo pipefail

# Cluster-side runner. It expects scripts/stage_offline_assets_from_s3.ps1 output
# to be copied to ASSET_ROOT before the job starts.

PYTHON_BIN="${PYTHON_BIN:-python}"
ASSET_ROOT="${ASSET_ROOT:-offline_assets/code-jepa-small}"
TOKENIZER_DIR="${TOKENIZER_DIR:-assets/tokenizers/codesearchnet-python/bpe16k}"
PRETRAIN_ROOT="${PRETRAIN_ROOT:-$ASSET_ROOT/pretrain/codesearchnet-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/equal-small-unixcoder-pretrain}"
STAGES="${STAGES:-transform-v0 transform-v1 transform-v2}"

STEPS="${STEPS:-200000}"
DURATION_HOURS="${DURATION_HOURS:-0}"
MAX_LEN="${MAX_LEN:-256}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-2e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
RANK_WEIGHT="${RANK_WEIGHT:-0.5}"
INBATCH_WEIGHT="${INBATCH_WEIGHT:-0.1}"
SIGREG_WEIGHT="${SIGREG_WEIGHT:-0.05}"
PRECISION="${PRECISION:-bf16}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
EVAL_BATCHES="${EVAL_BATCHES:-20}"
SEED="${SEED:-123456}"
PARALLEL="${PARALLEL:-1}"
CONTROL_GPU="${CONTROL_GPU:-0}"
JEPA_GPU="${JEPA_GPU:-1}"
MAX_SHARDS="${MAX_SHARDS:-0}"

mkdir -p "$OUTPUT_ROOT"

DATA_ROOTS=()
for stage in $STAGES; do
  root="$PRETRAIN_ROOT/$stage"
  if [[ ! -d "$root/views" || ! -d "$root/triples" ]]; then
    echo "Missing staged pretraining data under $root" >&2
    exit 2
  fi
  DATA_ROOTS+=("$root")
done

COMMON_ARGS=(
  scripts/train_codebert_jepa_torch.py
  --data-roots "${DATA_ROOTS[@]}"
  --model-name "$TOKENIZER_DIR"
  --init unixcoder_small_scratch
  --max-len "$MAX_LEN"
  --batch-size "$BATCH_SIZE"
  --steps "$STEPS"
  --duration-hours "$DURATION_HOURS"
  --lr "$LR"
  --warmup-steps "$WARMUP_STEPS"
  --inbatch-weight "$INBATCH_WEIGHT"
  --sigreg-weight "$SIGREG_WEIGHT"
  --precision "$PRECISION"
  --eval-every "$EVAL_EVERY"
  --eval-batches "$EVAL_BATCHES"
  --save-every "$SAVE_EVERY"
  --seed "$SEED"
)

if [[ "$MAX_SHARDS" != "0" ]]; then
  COMMON_ARGS+=(--max-shards "$MAX_SHARDS")
fi

run_control() {
  CUDA_VISIBLE_DEVICES="$CONTROL_GPU" "$PYTHON_BIN" "${COMMON_ARGS[@]}" \
    --output-dir "$OUTPUT_ROOT/control" \
    --rank-weight 0.0
}

run_jepa() {
  CUDA_VISIBLE_DEVICES="$JEPA_GPU" "$PYTHON_BIN" "${COMMON_ARGS[@]}" \
    --output-dir "$OUTPUT_ROOT/code_jepa" \
    --rank-weight "$RANK_WEIGHT"
}

echo "Pretraining control and Code-JEPA with identical data roots:"
printf '  %s\n' "${DATA_ROOTS[@]}"

if [[ "$PARALLEL" == "1" ]]; then
  run_control &
  control_pid=$!
  run_jepa &
  jepa_pid=$!
  status=0
  wait "$control_pid" || status=$?
  wait "$jepa_pid" || status=$?
  if [[ "$status" != "0" ]]; then
    exit "$status"
  fi
else
  run_control
  run_jepa
fi

echo "Done. Checkpoints:"
echo "  $OUTPUT_ROOT/control/latest.pt"
echo "  $OUTPUT_ROOT/code_jepa/latest.pt"
