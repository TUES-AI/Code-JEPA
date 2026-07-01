# Multi-GPU Siamese training plan

Default paper path: shared RoBERTa-style encoder + projection head, no unconditioned predictor.

## Data

```text
s3://code-jepa/tokenized/codesearchnet/bpe16k-buckets-128-256-512-1024-2048/
```

Cache contents: `59,404,781` triplets, `28` segments, `7,335` token shards, `138.2GB`.

## Default loss

```text
pos_loss        = 1 - cosine(z_anchor, z_pos)
rank_loss       = max(0, margin + sim(anchor, neg) - sim(anchor, pos))
inbatch_loss    = cross entropy over all devices' positives
sigreg_loss     = sliced Gaussian regularization on z_anchor/z_pos/z_neg
```

Recommended first-run coefficients:

```text
pos_weight      = 1.0
rank_weight     = 1.0
inbatch_weight  = 0.1
sigreg_weight   = 0.05
margin          = 0.2
temperature     = 0.05
```

Optimizer/schedule:

```text
AdamW
lr              = 3e-4
warmup_steps    = 1000
cosine decay to = 3e-5
weight_decay    = 0.01
grad_clip       = 1.0
precision       = bf16
dropout         = 0.0
```

## Bucket batch presets

Per-device batches in `scripts/train_siamese_bpe_jepa_multigpu.py`:

```text
h100: 128:512 256:512 512:128 1024:32 2048:8
a40:  128:256 256:256 512:64  1024:16 2048:4
safe: 128:128 256:128 512:32  1024:8  2048:2
```

RunPod 4x RTX PRO 4000 Blackwell 24GB result: the `a40` table is the best safe table. It improved 4-GPU scaling from `62.7%` to `78.0%`. Full `h100` table OOMed at bucket-256; hybrid larger long-bucket table later OOMed after several bucket shape compilations.

## Discoverer SLURM commands

```bash
sbatch slurm/profile-siamese-jax-multigpu.sh
sbatch slurm/pretrain-siamese-jax-multigpu.sh
```

Set `DATA_DIR` if the tokenized cache is staged somewhere other than:

```text
/valhalla/projects/bg-eng-01/scratch/code-jepa/tokenized/codesearchnet/bpe16k-buckets-128-256-512-1024-2048
```

## RunPod scaling profile command

Run this first on 2-4 A40/H100 devices. It executes the real multi-GPU trainer for each device count, then writes `scaling-summary.json` and `scaling-summary.md` with ETA, speedup, scaling efficiency, and incremental efficiency for each added GPU.

```bash
/opt/venv/bin/python scripts/profile_siamese_bpe_jepa.py \
  --data-dirs /proj/s3/tokenized/codesearchnet/bpe16k-buckets-128-256-512-1024-2048 \
  --output-dir /proj/s3/runs/bpe16k-siamese-a40-scaling-profile \
  --device-counts 1 2 4 \
  --model-size roberta_25m \
  --hardware-preset a40 \
  --max-len 2048 \
  --duration-minutes 5 \
  --log-every 10
```

Use `--hardware-preset h100` on H100 pods. For exact memory-ceiling tests, use `--hardware-preset custom --bucket-batches ...`.

## Direct trainer smoke command

```bash
/opt/venv/bin/python scripts/train_siamese_bpe_jepa_multigpu.py \
  --data-dirs /proj/s3/tokenized/codesearchnet/bpe16k-buckets-128-256-512-1024-2048 \
  --output-dir /proj/s3/runs/bpe16k-siamese-multigpu-smoke \
  --model-size roberta_25m \
  --hardware-preset h100 \
  --max-len 2048 \
  --duration-minutes 5 \
  --log-every 10 \
  --eval-every 0 \
  --save-every 1000 \
  --s3-output-prefix s3://code-jepa/runs/bpe16k-siamese-multigpu-smoke/
```
