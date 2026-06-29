#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=pretrain-unixcoder-raw
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -o /valhalla/projects/bg-eng-01/Code-JEPA/logs/pretrain-unixcoder-raw.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/Code-JEPA/logs/pretrain-unixcoder-raw.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

export VIRTUAL_ENV=/valhalla/projects/bg-eng-01/conda_envs/torch
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export HF_HOME=/valhalla/projects/bg-eng-01/scratch/vvasilev/hf-cache
export TOKENIZERS_PARALLELISM=false

PROJECT_DIR=/valhalla/projects/bg-eng-01/Code-JEPA

cd "${PROJECT_DIR}"
mkdir -p logs

python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| gpus", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this Slurm job")
print("GPU 0:", torch.cuda.get_device_name(0))
PY

OUTPUT_DIR=runs/pretrain-unixcoder-raw \
STEPS=40000 \
BATCH_SIZE=128 \
MAX_LEN=256 \
LR=2e-5 \
WARMUP_STEPS=2000 \
MLM_PROBABILITY=0.15 \
CONTRASTIVE_WEIGHT=0.1 \
PRECISION=bf16 \
SAVE_EVERY=5000 \
EVAL_EVERY=1000 \
EVAL_BATCHES=20 \
SEED=123456 \
GPU=0 \
bash scripts/run_raw_unixcoder_pretraining.sh

echo "Raw UniXcoder pretrain done: ${PROJECT_DIR}/runs/pretrain-unixcoder-raw/latest.pt"
