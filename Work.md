# Work

Identify who's coding agent you are by the harness.

## Toni - you are pi
1. Now
Best ready path is no-predictor JAX Siamese on all-language bpe16k cache with custom bucket batches `128:256 256:256 512:64 1024:16 2048:4`.

2. Next
On Discoverer, run `slurm/profile-siamese-jax-multigpu.sh`; if sane, run `slurm/pretrain-siamese-jax-multigpu.sh` for 1 epoch.

Notes
- Tokenized cache: 59,404,781 examples, 28 segments, 7,335 token shards, 7,392 S3 objects, 138.2GB.
- RTX PRO 4000 Blackwell 4-GPU profiling: safe table improved 4-GPU scaling from 62.7% to 78.0%; H100/hybrid larger tables OOMed on 24GB.
- Default trainer path is no-predictor Siamese: shared encoder + projection head; predictor only if later conditioned/ablated.
- Current active S3 data: `s3://code-jepa/data/codesearchnet/`, `s3://code-jepa/tokenizers/codesearchnet/`, and `s3://code-jepa/tokenized/codesearchnet/bpe16k-buckets-128-256-512-1024-2048/`.

## Vasko - you are codex/claude code
Do not use deleted `codesearchnet-python` paths. Use multilingual `s3://code-jepa/data/codesearchnet/`, tokenizer `s3://code-jepa/tokenizers/codesearchnet/bpe16k/`, and tokenized cache `s3://code-jepa/tokenized/codesearchnet/bpe16k-buckets-128-256-512-1024-2048/`.

