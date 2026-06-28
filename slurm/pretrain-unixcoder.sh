#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=pretrain-unixcoder
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/Code-JEPA/logs/pretrain-unixcoder.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/Code-JEPA/logs/pretrain-unixcoder.%j.err

set -euo pipefail
module purge
module load anaconda3
module load nvidia/cuda/12

export VIRTUAL_ENV=/valhalla/projects/bg-eng-01/conda_envs/torch
export PATH=${VIRTUAL_ENV}/bin:${PATH}

PROJECT_DIR=/valhalla/projects/bg-eng-01/Code-JEPA
DATA_ROOT=/valhalla/projects/bg-eng-01/scratch/vvasilev/code-jepa-small/pretrain/codesearchnet-python/transform-v0
OUTPUT_DIR=${PROJECT_DIR}/runs/pretrain-unixcoder

cd "${PROJECT_DIR}"
mkdir -p logs
pip install -e . --no-deps -q

torchrun --nproc_per_node=2 --master_port=29500 scripts/train_codebert_jepa_torch.py \
    --data-roots "${DATA_ROOT}" \
    --model-name assets/tokenizers/codesearchnet-python/bpe16k \
    --init unixcoder_small_scratch \
    --output-dir "${OUTPUT_DIR}" \
    --rank-weight 0.0 \
    --steps 40000 \
    --batch-size 128 \
    --max-len 256 \
    --lr 2e-5 \
    --warmup-steps 2000 \
    --inbatch-weight 0.1 \
    --sigreg-weight 0.05 \
    --precision bf16 \
    --save-every 5000 \
    --eval-every 1000 \
    --eval-batches 20 \
    --seed 123456

echo "UniXcoder pretrain done: ${OUTPUT_DIR}/latest.pt"
