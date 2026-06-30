#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=pretrain-unixcoder-core
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/Code-JEPA/logs/pretrain-unixcoder-core.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/Code-JEPA/logs/pretrain-unixcoder-core.%j.err

set -euo pipefail
module purge
module load anaconda3
module load nvidia/cuda/12

export VIRTUAL_ENV=/valhalla/projects/bg-eng-01/conda_envs/torch
export PATH=${VIRTUAL_ENV}/bin:${PATH}

PROJECT_DIR=/valhalla/projects/bg-eng-01/Code-JEPA
OUTPUT_DIR=${PROJECT_DIR}/runs/pretrain-unixcoder-core

cd "${PROJECT_DIR}"
mkdir -p logs
pip install -e . --no-deps -q

SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())" 2>/dev/null || echo "")
export SSL_CERT_FILE

# Smaller-scale UniXcoder reproduction (Guo et al. 2022, https://arxiv.org/abs/2203.03850):
# real MLM (15% masking, 80/10/10) + multimodal code-doc contrastive learning, mixed across
# the six CodeSearchNet languages with alpha=0.7 temperature sampling, same as the paper's
# language balancing. Scaled down to our 27M-param SmallUniXcoder (6L/512H) and 2-GPU DDP
# budget instead of the paper's 12L/768H model on 64 V100s for 8 days. This is the actual
# MLM+contrastive foundation our prior "control" baseline never had (it only ever trained
# with JEPA-style triple losses at rank_weight=0.0).
torchrun --nproc_per_node=2 --master_port=29700 scripts/train_small_unixcoder_raw.py \
    --output-dir "${OUTPUT_DIR}" \
    --model-name assets/tokenizers/codesearchnet/bpe16k \
    --dataset-name code_search_net \
    --languages python java javascript go php ruby \
    --language-alpha 0.7 \
    --split train \
    --max-len 256 \
    --batch-size 128 \
    --steps 40000 \
    --lr 2e-5 \
    --warmup-steps 2000 \
    --mlm-probability 0.15 \
    --contrastive-weight 0.1 \
    --temperature 0.05 \
    --precision bf16 \
    --save-every 5000 \
    --eval-every 1000 \
    --eval-batches 20 \
    --seed 123456

echo "UniXcoder core (MLM + code-doc contrastive) pretrain done: ${OUTPUT_DIR}/latest.pt"
