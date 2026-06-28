#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=code-jepa-v0-finetune
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/Code-JEPA/logs/v0-finetune.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/Code-JEPA/logs/v0-finetune.%j.err

set -euo pipefail
module purge
module load anaconda3
module load nvidia/cuda/12

export VIRTUAL_ENV=/valhalla/projects/bg-eng-01/conda_envs/torch
export PATH=${VIRTUAL_ENV}/bin:${PATH}

PROJECT_DIR=/valhalla/projects/bg-eng-01/Code-JEPA
ASSET_ROOT=/valhalla/projects/bg-eng-01/scratch/vvasilev/code-jepa-small
BENCH_ROOT=${ASSET_ROOT}/benchmarks/codexglue

CONTROL_CKPT=${PROJECT_DIR}/runs/v0-control-pretrain/control/latest.pt
JEPA_CKPT=${PROJECT_DIR}/runs/v0-jepa-pretrain/code_jepa/latest.pt

cd "${PROJECT_DIR}"
mkdir -p logs

pip install -e . --no-deps -q

# Verify checkpoints exist
for ckpt in "${CONTROL_CKPT}" "${JEPA_CKPT}"; do
    if [[ ! -f "${ckpt}" ]]; then
        echo "ERROR: missing checkpoint: ${ckpt}" >&2
        exit 1
    fi
    echo "checkpoint ok: ${ckpt}"
done

# Download benchmarks if needed
if [[ ! -d "${BENCH_ROOT}/poj104" || ! -d "${BENCH_ROOT}/bigclonebench" ]]; then
    echo "Downloading benchmarks..."
    SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())" 2>/dev/null || echo "")
    export SSL_CERT_FILE
    python scripts/download_codexglue_benchmarks.py \
        --output-root "${BENCH_ROOT}" \
        --benchmarks bigclonebench poj104 \
        --prepare-poj \
        --skip-existing
fi

# control -> GPU 0, code_jepa -> GPU 1, parallel per task
CONTROL_CKPT=${CONTROL_CKPT} \
JEPA_CKPT=${JEPA_CKPT} \
ASSET_ROOT=${ASSET_ROOT} \
OUTPUT_ROOT=runs/v0-finetune \
PARALLEL=1 \
CONTROL_GPU=0 \
JEPA_GPU=1 \
EPOCHS_POJ=2 \
EPOCHS_BCB=2 \
PRECISION=bf16 \
bash scripts/run_equal_finetuning.sh

echo "Fine-tune done. Results:"
for run in control code_jepa; do
    for task in poj104 bigclonebench; do
        f=runs/v0-finetune/${run}/${task}/results.json
        echo "=== ${run}/${task} ===" && cat "${f}"
    done
done
