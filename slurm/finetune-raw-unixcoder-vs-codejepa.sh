#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=ft-raw-vs-jepa
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/Code-JEPA/logs/ft-raw-vs-jepa.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/Code-JEPA/logs/ft-raw-vs-jepa.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

export VIRTUAL_ENV=/valhalla/projects/bg-eng-01/conda_envs/torch
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false

PROJECT_DIR=/valhalla/projects/bg-eng-01/Code-JEPA
ASSET_ROOT=/valhalla/projects/bg-eng-01/scratch/vvasilev/code-jepa-small
BENCH_ROOT=${ASSET_ROOT}/benchmarks/codexglue-hf
CONTROL_CKPT=runs/pretrain-unixcoder-raw/latest.pt
JEPA_CKPT=runs/pretrain-codejepa/latest.pt
OUTPUT_ROOT=runs/raw-unixcoder-vs-codejepa-finetune

cd "${PROJECT_DIR}"
mkdir -p logs

python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| gpus", torch.cuda.device_count())
if torch.cuda.device_count() < 2:
    raise RuntimeError("Expected 2 GPUs, got " + str(torch.cuda.device_count()))
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}:", torch.cuda.get_device_name(i))
PY

for path in "${CONTROL_CKPT}" "${JEPA_CKPT}"; do
    if [[ ! -f "${path}" ]]; then
        echo "ERROR: checkpoint not found: ${path}" >&2
        exit 1
    fi
done

if [[ ! -d "${BENCH_ROOT}/bigclonebench" ]]; then
    echo "ERROR: BigCloneBench not staged under ${BENCH_ROOT}/bigclonebench" >&2
    exit 1
fi

if [[ ! -d "${BENCH_ROOT}/poj104" ]]; then
    if [[ -d "${ASSET_ROOT}/benchmarks/codexglue/poj104" ]]; then
        ln -s "${ASSET_ROOT}/benchmarks/codexglue/poj104" "${BENCH_ROOT}/poj104"
    else
        echo "ERROR: POJ-104 not found under ${BENCH_ROOT}/poj104 or ${ASSET_ROOT}/benchmarks/codexglue/poj104" >&2
        exit 1
    fi
fi

BENCH_ROOT="${BENCH_ROOT}" \
ASSET_ROOT="${ASSET_ROOT}" \
CONTROL_CKPT="${CONTROL_CKPT}" \
JEPA_CKPT="${JEPA_CKPT}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
TASKS="poj104 bigclonebench" \
EPOCHS_POJ=2 \
EPOCHS_BCB=2 \
BATCH_SIZE=16 \
EVAL_BATCH_SIZE=64 \
PARALLEL=1 \
CONTROL_GPU=0 \
JEPA_GPU=1 \
bash scripts/run_equal_finetuning.sh

echo "Raw UniXcoder vs Code-JEPA fine-tune done. Results:"
find "${OUTPUT_ROOT}" -name results.json -print -exec cat {} \;
