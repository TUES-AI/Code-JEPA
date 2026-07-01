#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=profile-jax-siamese
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH -o /valhalla/projects/bg-eng-01/Code-JEPA/logs/profile-jax-siamese.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/Code-JEPA/logs/profile-jax-siamese.%j.err

set -euo pipefail
module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR=${PROJECT_DIR:-/valhalla/projects/bg-eng-01/Code-JEPA}
JAX_ENV=${JAX_ENV:-/valhalla/projects/bg-eng-01/conda_envs/jax}
DATA_DIR=${DATA_DIR:-/valhalla/projects/bg-eng-01/scratch/code-jepa/tokenized/codesearchnet/bpe16k-buckets-128-256-512-1024-2048}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/runs/profile-jax-siamese-${SLURM_JOB_ID}}
PYTHON_BIN=${PYTHON_BIN:-${JAX_ENV}/bin/python}

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

# Discoverer H200 path: use the high-memory preset first. If it OOMs, fall back
# to custom bucket batches `128:256 256:256 512:64 1024:16 2048:4` from 24GB RTX profiling.
"${PYTHON_BIN}" scripts/profile_siamese_bpe_jepa.py \
  --data-dirs "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --device-counts 1 2 4 \
  --model-size roberta_25m \
  --hardware-preset h200 \
  --max-len 2048 \
  --duration-minutes 5 \
  --log-every 10

echo "Profile done: ${OUTPUT_DIR}/scaling-summary.md"
