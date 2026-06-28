# Discoverer Small UniXcoder Runbook

This runbook is for the 25-30M parameter small UniXcoder-style Code-JEPA
experiment on Discoverer+. The tokenizer is committed in this repo; large data
is staged outside git and copied to project storage.

## What Gets Compared

Two matched pretraining runs:

```text
control    same model/data/steps, rank_weight=0.0
code_jepa  same model/data/steps, rank_weight=0.5 by default
```

Both use:

- committed tokenizer: `assets/tokenizers/codesearchnet-python/bpe16k`
- Python-only transformed Code-JEPA pretraining data: `transform-v0/v1/v2`
- same seed, batch size, sequence length, steps, SIGReg, and in-batch loss
- same downstream fine-tuning code for POJ-104 and BigCloneBench

Expected data transfer:

```text
Code-JEPA Python pretraining views/triples   ~2.86 GB
POJ-104 + BigCloneBench benchmarks           ~1.30 GB
Total practical transfer                     ~4.15 GB
Recommended free space                       10+ GB
```

## 1. Stage Data Locally

Run this on a machine with S3 access and public internet, not on the cluster.

Requirements:

```powershell
python -m pip install gdown
```

Then stage the offline data bundle:

```powershell
.\scripts\stage_offline_assets_from_s3.ps1 `
  -OutputDir offline_assets/code-jepa-small `
  -Stages transform-v0,transform-v1,transform-v2 `
  -ShardMode all `
  -SkipTokenizer `
  -IncludeBenchmarks
```

For a tiny transfer/smoke package:

```powershell
.\scripts\stage_offline_assets_from_s3.ps1 `
  -OutputDir offline_assets/code-jepa-small-smoke `
  -Stages transform-v0,transform-v1,transform-v2 `
  -ShardMode range `
  -ShardStart 0 `
  -ShardCount 2 `
  -SkipTokenizer `
  -IncludeBenchmarks
```

Do not commit `offline_assets/`; it is ignored by git.

## 2. Create Discoverer Project Directory

Set these placeholders:

```bash
export DISC_USER=<your_discoverer_username>
export PROJECT_ID=<your_slurm_account_or_project_id>
export REMOTE=login-plus.discoverer.bg
export PROJECT_ROOT=/valhalla/projects/$PROJECT_ID/scratch/$DISC_USER
```

Create the target directory:

```bash
ssh $DISC_USER@$REMOTE

mkdir -p /valhalla/projects/$PROJECT_ID/scratch/$USER/code-jepa-small
chmod 700 /valhalla/projects/$PROJECT_ID/scratch/$USER
exit
```

Use `/valhalla/projects/...`, not `$HOME`.

## 3. Transfer Data

Preferred transfer method is `rsync`, because it resumes interrupted transfers.

```bash
rsync -e "ssh" -avh --progress --partial --append \
  offline_assets/code-jepa-small/ \
  $DISC_USER@$REMOTE:/valhalla/projects/$PROJECT_ID/scratch/$DISC_USER/code-jepa-small/
```

For the smoke package:

```bash
rsync -e "ssh" -avh --progress --partial --append \
  offline_assets/code-jepa-small-smoke/ \
  $DISC_USER@$REMOTE:/valhalla/projects/$PROJECT_ID/scratch/$DISC_USER/code-jepa-small-smoke/
```

## 4. Clone Repo And Set Up Python On Cluster

On Discoverer+:

```bash
ssh $DISC_USER@$REMOTE

cd /valhalla/projects/$PROJECT_ID/scratch/$USER
git clone <your_repo_url> Code-JEPA
cd Code-JEPA

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[models]"
```

Verify CUDA:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

## 5. Smoke Pretraining

Run this before spending H200 time on the full job.

```bash
cd /valhalla/projects/$PROJECT_ID/scratch/$USER/Code-JEPA
source .venv/bin/activate
chmod +x scripts/run_equal_pretraining.sh scripts/run_equal_finetuning.sh

ASSET_ROOT=/valhalla/projects/$PROJECT_ID/scratch/$USER/code-jepa-small-smoke \
OUTPUT_ROOT=runs/smoke-pretrain \
STEPS=20 \
BATCH_SIZE=8 \
MAX_LEN=128 \
MAX_SHARDS=2 \
scripts/run_equal_pretraining.sh
```

Expected outputs:

```text
runs/smoke-pretrain/control/latest.pt
runs/smoke-pretrain/code_jepa/latest.pt
```

## 6. Full Equal Pretraining

This launches both runs in parallel by default:

```text
GPU 0 -> control
GPU 1 -> code_jepa
```

```bash
cd /valhalla/projects/$PROJECT_ID/scratch/$USER/Code-JEPA
source .venv/bin/activate

ASSET_ROOT=/valhalla/projects/$PROJECT_ID/scratch/$USER/code-jepa-small \
OUTPUT_ROOT=runs/equal-small-unixcoder-pretrain \
STEPS=200000 \
BATCH_SIZE=128 \
MAX_LEN=256 \
PRECISION=bf16 \
scripts/run_equal_pretraining.sh
```

Main knobs:

```bash
STEPS=200000
BATCH_SIZE=128
MAX_LEN=256
RANK_WEIGHT=0.5
PRECISION=bf16
PARALLEL=1
CONTROL_GPU=0
JEPA_GPU=1
```

Outputs:

```text
runs/equal-small-unixcoder-pretrain/control/latest.pt
runs/equal-small-unixcoder-pretrain/code_jepa/latest.pt
```

## 7. Smoke Fine-Tuning

```bash
ASSET_ROOT=/valhalla/projects/$PROJECT_ID/scratch/$USER/code-jepa-small \
PRETRAIN_OUTPUT_ROOT=runs/equal-small-unixcoder-pretrain \
OUTPUT_ROOT=runs/smoke-finetune \
MAX_TRAIN_EXAMPLES=256 \
MAX_VALID_EXAMPLES=256 \
MAX_TEST_EXAMPLES=256 \
EPOCHS_POJ=1 \
EPOCHS_BCB=1 \
scripts/run_equal_finetuning.sh
```

## 8. Full Equal Fine-Tuning

```bash
ASSET_ROOT=/valhalla/projects/$PROJECT_ID/scratch/$USER/code-jepa-small \
PRETRAIN_OUTPUT_ROOT=runs/equal-small-unixcoder-pretrain \
OUTPUT_ROOT=runs/equal-small-unixcoder-finetune \
scripts/run_equal_finetuning.sh
```

Outputs:

```text
runs/equal-small-unixcoder-finetune/control/poj104/results.json
runs/equal-small-unixcoder-finetune/code_jepa/poj104/results.json
runs/equal-small-unixcoder-finetune/control/bigclonebench/results.json
runs/equal-small-unixcoder-finetune/code_jepa/bigclonebench/results.json
```

## 9. Slurm Batch Skeleton

Adjust partition/account/time to your allocation.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=code-jepa-small-pretrain
#SBATCH --account=<your_slurm_account>
#SBATCH --partition=<gpu_partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

export PROJECT_ID=<your_slurm_account_or_project_id>
export WORK=/valhalla/projects/$PROJECT_ID/scratch/$USER/Code-JEPA
export ASSET_ROOT=/valhalla/projects/$PROJECT_ID/scratch/$USER/code-jepa-small

cd "$WORK"
mkdir -p logs
source .venv/bin/activate

OUTPUT_ROOT=runs/equal-small-unixcoder-pretrain \
STEPS=200000 \
BATCH_SIZE=128 \
MAX_LEN=256 \
PRECISION=bf16 \
scripts/run_equal_pretraining.sh
```

Submit:

```bash
sbatch slurm-code-jepa-small-pretrain.sh
```

Monitor:

```bash
squeue -u $USER
tail -f logs/code-jepa-small-pretrain-<jobid>.out
```

## 10. What Not To Do

- Do not put datasets in `$HOME`; Discoverer home quota is small.
- Do not commit `offline_assets/`, `runs/`, checkpoints, or parquet shards.
- Do not call this a full UniXcoder reproduction. It is a small UniXcoder-style
  architecture trained on Python-only Code-JEPA transformed data.
- Do not use Hugging Face CodeSearchNet as a drop-in pretraining replacement;
  it does not contain Code-JEPA `views/triples` hard-negative supervision.

