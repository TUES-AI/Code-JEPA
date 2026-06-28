#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=code-jepa-smoke-finetune
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/Code-JEPA/logs/smoke-finetune.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/Code-JEPA/logs/smoke-finetune.%j.err

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

ASSET_ROOT=/valhalla/projects/bg-eng-01/scratch/vvasilev/code-jepa-small \
PRETRAIN_OUTPUT_ROOT=runs/smoke-pretrain \
OUTPUT_ROOT=runs/smoke-finetune \
MAX_TRAIN_EXAMPLES=256 \
MAX_VALID_EXAMPLES=256 \
MAX_TEST_EXAMPLES=256 \
EPOCHS_POJ=1 \
EPOCHS_BCB=1 \
bash scripts/run_equal_finetuning.sh

echo "Smoke fine-tune done. Outputs:"
ls runs/smoke-finetune/control/ runs/smoke-finetune/code_jepa/
