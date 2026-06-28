#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=code-jepa-equal-pretrain
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/Code-JEPA/logs/equal-pretrain.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/Code-JEPA/logs/equal-pretrain.%j.err

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

ASSET_ROOT=/valhalla/projects/bg-eng-01/scratch/vvasilev/code-jepa-small \
OUTPUT_ROOT=runs/equal-small-unixcoder-pretrain \
STEPS=200000 \
BATCH_SIZE=128 \
MAX_LEN=256 \
PRECISION=bf16 \
bash scripts/run_equal_pretraining.sh

echo "Equal pretrain done. Checkpoints:"
ls runs/equal-small-unixcoder-pretrain/control/ runs/equal-small-unixcoder-pretrain/code_jepa/
