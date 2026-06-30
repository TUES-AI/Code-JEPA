#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=finetune-codejepa-v2
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/Code-JEPA/logs/finetune-codejepa-v2.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/Code-JEPA/logs/finetune-codejepa-v2.%j.err

set -euo pipefail
module purge
module load anaconda3
module load nvidia/cuda/12

export VIRTUAL_ENV=/valhalla/projects/bg-eng-01/conda_envs/torch
export PATH=${VIRTUAL_ENV}/bin:${PATH}

PROJECT_DIR=/valhalla/projects/bg-eng-01/Code-JEPA
CHECKPOINT=${PROJECT_DIR}/runs/pretrain-codejepa-v2/latest.pt
BENCH_ROOT=/valhalla/projects/bg-eng-01/scratch/vvasilev/code-jepa-small/benchmarks/codexglue
OUTPUT_DIR=${PROJECT_DIR}/runs/finetune-codejepa-v2

cd "${PROJECT_DIR}"
mkdir -p logs
pip install -e . --no-deps -q

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi
echo "Using checkpoint: ${CHECKPOINT}"

if [[ ! -d "${BENCH_ROOT}/poj104" || ! -d "${BENCH_ROOT}/bigclonebench" ]]; then
    echo "Downloading benchmarks..."
    SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())" 2>/dev/null || echo "")
    export SSL_CERT_FILE
    python scripts/download_codexglue_benchmarks.py \
        --output-root "${BENCH_ROOT}" \
        --benchmarks bigclonebench poj104 \
        --prepare-poj --skip-existing
fi

echo "=== Fine-tuning Code-JEPA (v2) on POJ-104 (2 GPUs) ==="
torchrun --nproc_per_node=2 --master_port=29611 scripts/finetune_clone_benchmarks.py \
    --benchmark poj104 \
    --benchmark-dir "${BENCH_ROOT}/poj104" \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}/poj104" \
    --model-name assets/tokenizers/codesearchnet-python/bpe16k \
    --max-len 256 --batch-size 32 --eval-batch-size 64 \
    --epochs 2 --lr 2e-5 --head-lr 1e-4 --precision bf16 --seed 123456

echo "=== Fine-tuning Code-JEPA (v2) on BigCloneBench (2 GPUs) ==="
torchrun --nproc_per_node=2 --master_port=29611 scripts/finetune_clone_benchmarks.py \
    --benchmark bigclonebench \
    --benchmark-dir "${BENCH_ROOT}/bigclonebench" \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}/bigclonebench" \
    --model-name assets/tokenizers/codesearchnet-python/bpe16k \
    --max-len 256 --batch-size 32 --eval-batch-size 64 \
    --epochs 2 --lr 2e-5 --head-lr 1e-4 --precision bf16 --seed 123456

echo "Code-JEPA (v2) fine-tune done. Results:"
for task in poj104 bigclonebench; do
    echo "=== ${task} ===" && cat "${OUTPUT_DIR}/${task}/results.json"
done
