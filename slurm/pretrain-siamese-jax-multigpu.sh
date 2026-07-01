#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=pretrain-jax-siamese
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/Code-JEPA/logs/pretrain-jax-siamese.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/Code-JEPA/logs/pretrain-jax-siamese.%j.err

set -euo pipefail
module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR=${PROJECT_DIR:-/valhalla/projects/bg-eng-01/Code-JEPA}
JAX_ENV=${JAX_ENV:-/valhalla/projects/bg-eng-01/conda_envs/torch}
DATA_DIR=${DATA_DIR:-/valhalla/projects/bg-eng-01/scratch/code-jepa/tokenized/codesearchnet/bpe16k-buckets-128-256-512-1024-2048}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/runs/pretrain-jax-siamese-${SLURM_JOB_ID}}
PYTHON_BIN=${PYTHON_BIN:-${JAX_ENV}/bin/python}
STOP_AFTER_EPOCHS=${STOP_AFTER_EPOCHS:-1.0}

export VIRTUAL_ENV=${JAX_ENV}
export PATH=${VIRTUAL_ENV}/bin:${PATH}
export TF_GPU_ALLOCATOR=${TF_GPU_ALLOCATOR:-cuda_malloc_async}

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

if [ ! -d "${DATA_DIR}" ]; then
  echo "Missing DATA_DIR=${DATA_DIR}" >&2
  echo "Set DATA_DIR to the local copy of s3://code-jepa/tokenized/codesearchnet/bpe16k-buckets-128-256-512-1024-2048/" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import jax
print("jax", jax.__version__, "backend", jax.default_backend(), "devices", len(jax.devices()))
assert jax.default_backend() == "gpu"
PY

"${PYTHON_BIN}" -m pip install -e . --no-deps -q

"${PYTHON_BIN}" scripts/train_siamese_bpe_jepa_multigpu.py \
  --data-dirs "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-devices 2 \
  --model-size roberta_25m \
  --hardware-preset h200 \
  --max-len 2048 \
  --steps 1000000 \
  --target-epochs "${STOP_AFTER_EPOCHS}" \
  --stop-after-epochs "${STOP_AFTER_EPOCHS}" \
  --lr 3e-4 \
  --warmup-steps 1000 \
  --end-lr-ratio 0.1 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --pos-weight 1.0 \
  --rank-weight 1.0 \
  --inbatch-weight 0.1 \
  --sigreg-weight 0.05 \
  --margin 0.2 \
  --temperature 0.05 \
  --precision bf16 \
  --log-every 20 \
  --eval-every 0 \
  --save-every 1000 \
  --s3-sync-every 0

echo "Pretrain done: ${OUTPUT_DIR}/latest.pkl"
