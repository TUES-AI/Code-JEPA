#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=code-jepa-smoke-pretrain
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/Code-JEPA/logs/smoke-pretrain.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/Code-JEPA/logs/smoke-pretrain.%j.err

set -euo pipefail
module purge
module load anaconda3
module load nvidia/cuda/12

export VIRTUAL_ENV=/valhalla/projects/bg-eng-01/conda_envs/torch
export PATH=${VIRTUAL_ENV}/bin:${PATH}

PROJECT_DIR=/valhalla/projects/bg-eng-01/Code-JEPA
cd "${PROJECT_DIR}"
mkdir -p logs

pip install -e . --no-deps -q

python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| gpus", torch.cuda.device_count())
PY

ASSET_ROOT=/valhalla/projects/bg-eng-01/scratch/vvasilev/code-jepa-small-smoke \
OUTPUT_ROOT=runs/smoke-pretrain \
STEPS=20 \
BATCH_SIZE=8 \
MAX_LEN=128 \
MAX_SHARDS=2 \
PRECISION=bf16 \
bash scripts/run_equal_pretraining.sh

echo "Smoke pretrain done. Outputs:"
ls runs/smoke-pretrain/control/ runs/smoke-pretrain/code_jepa/
