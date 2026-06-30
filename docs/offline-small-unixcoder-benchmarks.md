# Offline Small UniXcoder Benchmarks

This runbook stages large data while S3 and public internet are available, then
runs the 25-30M parameter small UniXcoder comparison on an offline cluster. The
small `bpe16k` tokenizer is committed in the repo under
`assets/tokenizers/codesearchnet-python/bpe16k`.

## 1. Stage Assets

Run on a machine that can reach the Code-JEPA S3 bucket and the public
CodeXGLUE downloads:

```powershell
.\scripts\stage_offline_assets_from_s3.ps1 `
  -OutputDir offline_assets/code-jepa-small `
  -Stages transform-v0,transform-v1,transform-v2 `
  -ShardMode all `
  -SkipTokenizer `
  -IncludeBenchmarks
```

For a fast smoke package, replace `-ShardMode all` with:

```powershell
-ShardMode range -ShardStart 0 -ShardCount 2
```

Copy `offline_assets/code-jepa-small` to the cluster. After this copy, the
cluster run does not need S3. The tokenizer comes from the repo:

```text
assets/tokenizers/codesearchnet-python/bpe16k
```

## 2. Equal Pretraining

On the cluster, from the repo root:

```bash
ASSET_ROOT=/path/to/offline_assets/code-jepa-small \
OUTPUT_ROOT=runs/equal-small-unixcoder-pretrain \
STEPS=200000 \
BATCH_SIZE=128 \
MAX_LEN=256 \
scripts/run_equal_pretraining.sh
```

This launches two matched runs by default:

- `control`: same model, tokenizer, data roots, steps, and seed, with
  `rank_weight=0.0`.
- `code_jepa`: same settings, with hard-negative Code-JEPA ranking enabled.

The checkpoints land at:

```text
runs/equal-small-unixcoder-pretrain/control/latest.pt
runs/equal-small-unixcoder-pretrain/code_jepa/latest.pt
```

## 3. Equal Fine-Tuning

```bash
ASSET_ROOT=/path/to/offline_assets/code-jepa-small \
PRETRAIN_OUTPUT_ROOT=runs/equal-small-unixcoder-pretrain \
OUTPUT_ROOT=runs/equal-small-unixcoder-finetune \
scripts/run_equal_finetuning.sh
```

The fine-tuning script uses the same data caps, epochs, max length, batch size,
learning rates, and seed for both checkpoints. `MAX_*_EXAMPLES=0` means full
official splits.

Outputs:

```text
runs/equal-small-unixcoder-finetune/control/poj104/results.json
runs/equal-small-unixcoder-finetune/code_jepa/poj104/results.json
runs/equal-small-unixcoder-finetune/control/bigclonebench/results.json
runs/equal-small-unixcoder-finetune/code_jepa/bigclonebench/results.json
```

## Equality Contract

- Pretraining data roots are identical for both runs.
- Pretraining step count, seed, tokenizer, sequence length, batch size, SIGReg,
  and in-batch positive objective are identical.
- The only pretraining objective change is hard-negative rank loss:
  `rank_weight=0.0` versus `rank_weight=$RANK_WEIGHT`.
- Fine-tuning uses one shared implementation for both checkpoints.
- POJ-104 reports MAP@R. BigCloneBench reports binary metrics and applies the
  validation-selected threshold to test scores.
