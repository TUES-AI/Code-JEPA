#!/usr/bin/env bash
set -euo pipefail

# Fine-tune both equal-pretrained checkpoints on POJ-104 and BigCloneBench.

PYTHON_BIN="${PYTHON_BIN:-python}"
ASSET_ROOT="${ASSET_ROOT:-offline_assets/code-jepa-small}"
TOKENIZER_DIR="${TOKENIZER_DIR:-assets/tokenizers/codesearchnet-python/bpe16k}"
BENCH_ROOT="${BENCH_ROOT:-$ASSET_ROOT/benchmarks/codexglue}"
PRETRAIN_OUTPUT_ROOT="${PRETRAIN_OUTPUT_ROOT:-runs/equal-small-unixcoder-pretrain}"
CONTROL_CKPT="${CONTROL_CKPT:-$PRETRAIN_OUTPUT_ROOT/control/latest.pt}"
JEPA_CKPT="${JEPA_CKPT:-$PRETRAIN_OUTPUT_ROOT/code_jepa/latest.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/equal-small-unixcoder-finetune}"
TASKS="${TASKS:-poj104 bigclonebench}"

MAX_LEN="${MAX_LEN:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
EPOCHS_POJ="${EPOCHS_POJ:-2}"
EPOCHS_BCB="${EPOCHS_BCB:-2}"
LR="${LR:-2e-5}"
HEAD_LR="${HEAD_LR:-1e-4}"
PRECISION="${PRECISION:-bf16}"
SEED="${SEED:-123456}"
CONTROL_GPU="${CONTROL_GPU:-0}"
JEPA_GPU="${JEPA_GPU:-1}"
PARALLEL="${PARALLEL:-1}"

# Zero means use the full official split. Set small caps for smoke tests.
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-0}"
MAX_VALID_EXAMPLES="${MAX_VALID_EXAMPLES:-0}"
MAX_TEST_EXAMPLES="${MAX_TEST_EXAMPLES:-0}"

mkdir -p "$OUTPUT_ROOT"

run_one() {
  local name="$1"
  local checkpoint="$2"
  local gpu="$3"
  local task="$4"
  local task_dir="$5"
  local epochs="$6"

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/finetune_clone_benchmarks.py \
    --benchmark "$task" \
    --benchmark-dir "$task_dir" \
    --checkpoint "$checkpoint" \
    --output-dir "$OUTPUT_ROOT/$name/$task" \
    --model-name "$TOKENIZER_DIR" \
    --max-len "$MAX_LEN" \
    --batch-size "$BATCH_SIZE" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --epochs "$epochs" \
    --lr "$LR" \
    --head-lr "$HEAD_LR" \
    --precision "$PRECISION" \
    --seed "$SEED" \
    --max-train-examples "$MAX_TRAIN_EXAMPLES" \
    --max-valid-examples "$MAX_VALID_EXAMPLES" \
    --max-test-examples "$MAX_TEST_EXAMPLES"
}

for task in $TASKS; do
  case "$task" in
    poj104)
      task_dir="$BENCH_ROOT/poj104"
      epochs="$EPOCHS_POJ"
      ;;
    bigclonebench)
      task_dir="$BENCH_ROOT/bigclonebench"
      epochs="$EPOCHS_BCB"
      ;;
    *)
      echo "Unknown task: $task" >&2
      exit 2
      ;;
  esac

  if [[ ! -d "$task_dir" ]]; then
    echo "Missing benchmark directory: $task_dir" >&2
    exit 2
  fi

  echo "Fine-tuning $task with identical settings for control and Code-JEPA"
  if [[ "$PARALLEL" == "1" ]]; then
    run_one control "$CONTROL_CKPT" "$CONTROL_GPU" "$task" "$task_dir" "$epochs" &
    control_pid=$!
    run_one code_jepa "$JEPA_CKPT" "$JEPA_GPU" "$task" "$task_dir" "$epochs" &
    jepa_pid=$!
    status=0
    wait "$control_pid" || status=$?
    wait "$jepa_pid" || status=$?
    if [[ "$status" != "0" ]]; then
      exit "$status"
    fi
  else
    run_one control "$CONTROL_CKPT" "$CONTROL_GPU" "$task" "$task_dir" "$epochs"
    run_one code_jepa "$JEPA_CKPT" "$JEPA_GPU" "$task" "$task_dir" "$epochs"
  fi
done

echo "Done. Results are under $OUTPUT_ROOT/{control,code_jepa}/{poj104,bigclonebench}/results.json"
