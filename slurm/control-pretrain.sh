#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=code-jepa-control
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -o /valhalla/projects/bg-eng-01/Code-JEPA/logs/control-pretrain.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/Code-JEPA/logs/control-pretrain.%j.err

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
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

ASSET_ROOT=/valhalla/projects/bg-eng-01/scratch/vvasilev/code-jepa-small \
OUTPUT_ROOT=runs/v0-control-pretrain \
STAGES=transform-v0 \
STEPS=40000 \
BATCH_SIZE=128 \
MAX_LEN=256 \
PRECISION=bf16 \
PARALLEL=0 \
CONTROL_GPU=0 \
bash scripts/run_equal_pretraining.sh

echo "Control pretrain done."
ls runs/v0-control-pretrain/control/
