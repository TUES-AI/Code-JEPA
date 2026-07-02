#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=finetune-jax-siamese
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o /valhalla/projects/bg-eng-01/Code-JEPA/logs/finetune-jax-siamese.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/Code-JEPA/logs/finetune-jax-siamese.%j.err

set -euo pipefail
module purge
module load anaconda3

PROJECT_DIR=${PROJECT_DIR:-/valhalla/projects/bg-eng-01/Code-JEPA}
JAX_ENV=${JAX_ENV:-/valhalla/projects/bg-eng-01/conda_envs/torch}
PYTHON_BIN=${PYTHON_BIN:-${JAX_ENV}/bin/python}
CHECKPOINT=${CHECKPOINT:-${PROJECT_DIR}/runs/pretrain-jax-siamese-177664/latest.pkl}
TOKENIZER=${TOKENIZER:-${PROJECT_DIR}/assets/tokenizers/codesearchnet/bpe16k}
BENCH_ROOT=${BENCH_ROOT:-/valhalla/projects/bg-eng-01/scratch/vvasilev/code-jepa-small/benchmarks/codexglue}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/runs/finetune-jax-siamese-${SLURM_JOB_ID}}
EMBEDDING=${EMBEDDING:-z}

export VIRTUAL_ENV=${JAX_ENV}
export PATH=${VIRTUAL_ENV}/bin:${PATH}
export TF_GPU_ALLOCATOR=${TF_GPU_ALLOCATOR:-cuda_malloc_async}

# Point to pip-bundled CUDA libs so JAX doesn't pick up conflicting system CUDA
SITE_PACKAGES=$("${PYTHON_BIN}" -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="${SITE_PACKAGES}/nvidia/cusparse/lib:${SITE_PACKAGES}/nvidia/cublas/lib:${SITE_PACKAGES}/nvidia/cuda_runtime/lib:${SITE_PACKAGES}/nvidia/cudnn/lib:${SITE_PACKAGES}/nvidia/nccl/lib:${SITE_PACKAGES}/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

if [ ! -f "${CHECKPOINT}" ]; then
  echo "Missing CHECKPOINT=${CHECKPOINT}" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import jax
print("jax", jax.__version__, "backend", jax.default_backend(), "devices", len(jax.devices()))
assert jax.default_backend() == "gpu"
PY

"${PYTHON_BIN}" -m pip install -e . --no-deps -q

if [ ! -d "${BENCH_ROOT}/poj104" ] || [ ! -d "${BENCH_ROOT}/bigclonebench" ]; then
  echo "Downloading benchmarks..."
  SSL_CERT_FILE=$("${PYTHON_BIN}" -c "import certifi; print(certifi.where())" 2>/dev/null || echo "")
  export SSL_CERT_FILE
  "${PYTHON_BIN}" scripts/download_codexglue_benchmarks.py \
    --output-root "${BENCH_ROOT}" \
    --benchmarks bigclonebench poj104 \
    --prepare-poj --skip-existing
fi

echo "=== Fine-tuning (JAX) on POJ-104 [embedding=${EMBEDDING}] ==="
"${PYTHON_BIN}" scripts/finetune_clone_benchmarks_jax.py \
  --benchmark poj104 \
  --benchmark-dir "${BENCH_ROOT}/poj104" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}/poj104" \
  --tokenizer-path "${TOKENIZER}" \
  --embedding "${EMBEDDING}" \
  --max-len 256 --batch-size 32 --eval-batch-size 128 \
  --epochs 2 --lr 2e-5 --head-lr 1e-4 --precision bf16 --seed 123456

echo "=== Fine-tuning (JAX) on BigCloneBench [embedding=${EMBEDDING}] ==="
"${PYTHON_BIN}" scripts/finetune_clone_benchmarks_jax.py \
  --benchmark bigclonebench \
  --benchmark-dir "${BENCH_ROOT}/bigclonebench" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}/bigclonebench" \
  --tokenizer-path "${TOKENIZER}" \
  --embedding "${EMBEDDING}" \
  --max-len 256 --batch-size 32 --eval-batch-size 128 \
  --epochs 2 --lr 2e-5 --head-lr 1e-4 --precision bf16 --seed 123456 \
  --max-train-examples 100000

echo "Fine-tune done. Results:"
for task in poj104 bigclonebench; do
  echo "=== ${task} ===" && cat "${OUTPUT_DIR}/${task}/results.json"
done
