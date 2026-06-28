# A6000 Training Profile — 2026-06-27

## Setup

- GPU: NVIDIA RTX A6000, 48 GB
- Main path: `src/code_jepa/training/siamese_bpe_jepa.py`
- CLI: `scripts/train_siamese_bpe_jepa.py`
- Data: tokenized BPE16k cache `bpe16k-v0-perf262k-20260627-124558`
- Model: 6 layers, width 512, 8 heads, MLP 2048, max length 256, bf16

## Main result

Fixed-length 256 training is GPU-compute bound, not data/input bound.

Best profiled fixed-256 setting:

```text
batch size: 320
throughput: ~491 triples/s
train step: ~648 ms
forward only: ~154 ms
train/forward ratio: ~4.2x
```

GPU telemetry during active training:

```text
SM utilization: ~96%
memory utilization: ~69%
power: ~272 W
peak memory: ~36.7 GB
PCIe traffic: negligible
```

Segment timing:

```text
loader: ~0.5 ms/step
host->device copy: ~1.1 ms/step
training compute: ~648 ms/step
```

HLO dump confirmed `jax.nn.dot_product_attention(..., implementation="cudnn")` lowered to cuDNN FlashAttention custom calls.

## Scaling probes

Sequence length at batch 224:

```text
L=128: ~815 triples/s
L=192: ~605 triples/s
L=256: ~484-491 triples/s
```

Layer count at length 256, batch 224:

```text
2 layers: ~1170 triples/s
4 layers: ~703 triples/s
6 layers: ~491 triples/s
8 layers: ~384 triples/s
```

MLP width at 6 layers, length 256, batch 224:

```text
MLP 1024: ~610 triples/s
MLP 2048: ~490 triples/s
MLP 4096: ~390 triples/s
```

Head count had little effect between 4 and 8 heads; 16 heads was slower.

## Length distribution in current tokenized cache

```text
mean max(view length): ~171 tokens
<=128 tokens: ~36%
<=192 tokens: ~54%
truncated/at 256: ~34%
```

Expected gain from length bucketing, using measured L=128/192/256 speeds: about **1.2x** over fixed 256.

## Conclusion

Do not rent H100/H200 for this exact path yet. The current implementation is already using cuDNN FlashAttention and saturates the A6000. The biggest near-term win is length bucketing / shorter-sequence caches, then model-size tradeoffs such as MLP width or layer count.

Profile index:

```text
/Volumes/SSD/datasets/code-jepa/perf/profile-index.json
s3://code-jepa/perf/profile-index.json
```
