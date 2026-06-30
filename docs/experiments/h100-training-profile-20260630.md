# H100 training profile - 2026-06-30

Hardware: RunPod H100 PCIe, CUDA 12.8 driver path.

## Smoke setup

- model: `roberta_25m` (`25,958,016` params)
- tokenizer: `bpe16k`
- data: Python `v0+v1+v2` core from multilingual CodeSearchNet
- tokenized cache: `600,000` triplets, `max_len=256`, batch size `512`
- run stopped early after enough throughput signal

## Result

Median after warmup:

- `1,691` examples/s
- `1.30M` padded tokens/s
- `0.099` hours/epoch on the 600k-example smoke cache

Training reached step `750`; late rank accuracy was commonly `82%-94%` on train-distribution synthetic triples.

Artifact path:

```text
s3://code-jepa/runs/jepa-python-h100-smoke/jepa-python-h100-smoke-20260630-171552/
```

## Projection

Full prepared multilingual dataset has `59,404,781` triplets.

At measured `1,691` examples/s:

- 1 epoch: about `9.8` hours
- cost at `$3/h`: about `$29-$30`

This assumes fixed `max_len=256`; bucketed sequence lengths should reduce the compute bill for shorter triplets.

## Length note

Sampled raw BPE view lengths before truncation across six languages/core stages:

- view mean: `221`
- view median: `128`
- view p90: `466`
- view p95: `668`
- view p99: `1418`
- views over 256: `24.8%`

This supports bucketed tokenized caches such as `128/256/512/1024/2048` rather than one global padded length.
