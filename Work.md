# Work

Identify who's coding agent you are by the harness.

## Toni - you are pi
1. Now
All-language bpe16k bucketed tokenized cache is on S3: `s3://code-jepa/tokenized/codesearchnet/bpe16k-buckets-128-256-512-1024-2048/`.

2. Next
Run a short H100 smoke on that cache, then start the first roberta_25m epoch if throughput is acceptable.

Notes
- Tokenized cache: 59,404,781 examples, 28 segments, 7,335 token shards, 7,392 S3 objects, 138.2GB.
- H100 fixed-256 smoke artifacts: `s3://code-jepa/runs/jepa-python-h100-smoke/jepa-python-h100-smoke-20260630-171552/`.
- Current active S3 data: `s3://code-jepa/data/codesearchnet/`, `s3://code-jepa/tokenizers/codesearchnet/`, and `s3://code-jepa/tokenized/codesearchnet/bpe16k-buckets-128-256-512-1024-2048/`.

## Vasko - you are codex/claude code
Do not use deleted `codesearchnet-python` paths. Use multilingual `s3://code-jepa/data/codesearchnet/`, tokenizer `s3://code-jepa/tokenizers/codesearchnet/bpe16k/`, and tokenized cache `s3://code-jepa/tokenized/codesearchnet/bpe16k-buckets-128-256-512-1024-2048/`.

