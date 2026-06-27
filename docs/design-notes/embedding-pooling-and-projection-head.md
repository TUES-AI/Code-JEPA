# Embedding pooling and projection-head decision

Date: 2026-06-27

## Decision

For the main Code-JEPA encoder path, use mask-aware mean pooling over transformer hidden states to produce the reusable representation `h`, then use a small projection head to produce the training-space vector `z`:

```text
hidden states H -> mask-aware mean pool -> h
h -> Dense(8H) -> split 4H/4H -> SwiGLU -> RMSNorm -> Dense(D) -> z
```

Train JEPA prediction, ranking, and SIGReg on `z`. Keep/export `h` as the downstream/search embedding candidate.

## Rationale

- Mask-aware mean pooling is the safest default for bidirectional encoder embeddings because it aggregates all real tokens while ignoring padding.
- Raw CLS should be treated as an ablation, not the default, unless training explicitly makes CLS carry the sequence representation.
- Sum pooling should be avoided because embedding norm becomes sequence-length dependent, which can interact badly with SIGReg.
- Applying SIGReg in projected space lets the regularizer shape `z` without forcing the reusable encoder embedding `h` to be exactly Gaussian.

## Current code status

The main BPE path implements this projection-head setup:

- core: `src/code_jepa/training/siamese_bpe_jepa.py`
- CLI: `scripts/train_siamese_bpe_jepa.py`
- token cache builder: `scripts/tokenize_jepa_triples.py`

Current pooling excludes `<pad>` but includes `<bos>` and `<eos>` because the mask is `token_id != pad_token_id`.

## Next ablations

1. Current projection-head path: `Dense(8H) -> SwiGLU -> RMSNorm -> Dense(D)`.
2. No-head pooled-output baseline for comparison only.
3. Optional pooling ablation: include vs exclude `<bos>/<eos>` from the mean.
4. Optional CLS ablation only if a CLS-style token is explicitly trained/evaluated.
