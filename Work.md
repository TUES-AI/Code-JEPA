# Work

Identify who's coding agent you are by the harness.

## Toni - you are pi
1. Now
Moved the performant Siamese BPE Code-JEPA implementation into `src/code_jepa/training/siamese_bpe_jepa.py`; `scripts/train_siamese_bpe_jepa.py` is now only a CLI wrapper. The path uses projected `z` with `Dense(8H) -> SwiGLU -> RMSNorm -> Dense(D)` and requests cuDNN dot-product attention through `MultiHeadDotProductAttention(attention_fn=...)`.

2. Next
Run a short GPU smoke for the RMSNorm + cuDNN attention + projection-head path, then compare throughput and early rank behavior against `jax-tokenized-b224-blocking-20260627-141926`.

## Vasko - you are codex/claude code

