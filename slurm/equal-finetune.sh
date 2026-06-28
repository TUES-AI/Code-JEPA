#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=code-jepa-equal-finetune
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/Code-JEPA/logs/equal-finetune.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/Code-JEPA/logs/equal-finetune.%j.err

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
PRETRAIN_OUTPUT_ROOT=runs/equal-small-unixcoder-pretrain \
OUTPUT_ROOT=runs/equal-small-unixcoder-finetune \
bash scripts/run_equal_finetuning.sh

echo "Equal fine-tune done. Results:"
cat runs/equal-small-unixcoder-finetune/control/poj104/results.json
cat runs/equal-small-unixcoder-finetune/code_jepa/poj104/results.json
cat runs/equal-small-unixcoder-finetune/control/bigclonebench/results.json
cat runs/equal-small-unixcoder-finetune/code_jepa/bigclonebench/results.json
