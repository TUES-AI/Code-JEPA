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
ASSET_ROOT=/valhalla/projects/bg-eng-01/scratch/vvasilev/code-jepa-small
BENCH_ROOT=${ASSET_ROOT}/benchmarks/codexglue

cd "${PROJECT_DIR}"
mkdir -p logs

pip install -e . --no-deps -q

python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| gpus", torch.cuda.device_count())
if torch.cuda.device_count() < 2:
    raise RuntimeError("Expected 2 GPUs, got " + str(torch.cuda.device_count()))
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}:", torch.cuda.get_device_name(i))
PY

# Download benchmarks if not already present (needs internet; login-plus has it)
if [[ ! -d "${BENCH_ROOT}/poj104" || ! -d "${BENCH_ROOT}/bigclonebench" ]]; then
    echo "Downloading CodeXGLUE benchmarks..."
    SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())" 2>/dev/null || echo "")
    export SSL_CERT_FILE
    python scripts/download_codexglue_benchmarks.py \
        --output-root "${BENCH_ROOT}" \
        --benchmarks bigclonebench poj104 \
        --prepare-poj \
        --skip-existing
else
    echo "Benchmarks already present at ${BENCH_ROOT}"
fi

# control -> GPU 0, code_jepa -> GPU 1, both run in parallel per task
ASSET_ROOT=${ASSET_ROOT} \
PRETRAIN_OUTPUT_ROOT=runs/equal-small-unixcoder-pretrain \
OUTPUT_ROOT=runs/equal-small-unixcoder-finetune \
PARALLEL=1 \
CONTROL_GPU=0 \
JEPA_GPU=1 \
bash scripts/run_equal_finetuning.sh

echo "Equal fine-tune done. Results:"
for run in control code_jepa; do
    for task in poj104 bigclonebench; do
        f=runs/equal-small-unixcoder-finetune/${run}/${task}/results.json
        echo "=== ${run}/${task} ===" && cat "${f}"
    done
done
