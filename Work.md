# Work

Identify who's coding agent you are by the harness.

## Toni - you are pi
1. Now
H100 smoke is done and stopped. Fixed `max_len=256` roberta_25m measured ~1,691 examples/s / ~1.30M padded tokens/s; full 59.4M triplet epoch projects to ~9.8h / ~$30 at $3/h.

2. Next
Use bucketed tokenized caches (`128/256/512/1024/2048`) before any full run; rerun a short H100 smoke on the bucketed cache.

Notes
- Current committed path includes `scripts/tokenize_jepa_triples.py`, `src/code_jepa/training/siamese_bpe_jepa.py`, and `docs/experiments/h100-training-profile-20260630.md`.
- Bucketed tokenization/training is implemented and locally smoke-tested: tokenizer writes `bucket-XXXX/shard-*.npz`; trainer supports variable-length shards with one JIT compile per bucket length.
- H100 run artifacts: `s3://code-jepa/runs/jepa-python-h100-smoke/jepa-python-h100-smoke-20260630-171552/`.
- Current active S3 data: `s3://code-jepa/data/codesearchnet/` and `s3://code-jepa/tokenizers/codesearchnet/`.

## Vasko - you are codex/claude code
Do not use deleted `codesearchnet-python` paths. Use multilingual `s3://code-jepa/data/codesearchnet/` and tokenizer `s3://code-jepa/tokenizers/codesearchnet/bpe16k/`; prefer bucketed tokenized caches over fixed 256 padding.

