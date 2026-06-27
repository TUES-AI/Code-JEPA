#!/usr/bin/env python3
"""Build tokenized Code-JEPA triple shards for high-throughput training.

Input: prepared Parquet roots with matched `views/` and `triples/` shards.
Output: fixed-shape `.npz` shards containing uint16 token ids:

    tokens[example, view, token] where view = 0 anchor, 1 positive, 2 negative

The tokenized format is intentionally simple for fast single-GPU JAX training:
- no raw strings in the training loop;
- no view-id joins during training;
- masks are implicit (`tokens != pad_token_id`);
- transform ids are stored for evaluation/debugging.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast


@dataclass(frozen=True)
class Config:
    input_roots: list[str]
    tokenizer_path: str
    output_dir: str
    max_len: int = 256
    max_examples: int | None = None
    max_shards_per_root: int | None = None
    output_shard_size: int = 8192
    tokenize_batch_size: int = 1024
    seed: int = 0
    s3_output_prefix: str = ""


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-roots", nargs="+", required=True)
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--max-shards-per-root", type=int, default=None)
    p.add_argument("--output-shard-size", type=int, default=8192)
    p.add_argument("--tokenize-batch-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--s3-output-prefix", default="")
    return Config(**vars(p.parse_args()))


def main() -> None:
    cfg = parse_args()
    out = Path(cfg.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.tokenizer_path)
    pad_id = int(tokenizer.pad_token_id or 0)
    if len(tokenizer) > np.iinfo(np.uint16).max:
        raise ValueError(f"vocab size {len(tokenizer)} does not fit uint16 token cache")

    transform_vocab: dict[str, dict[str, int]] = {"positive": {}, "negative": {}, "negative_type": {}}
    manifest: dict[str, Any] = {
        "format": "code-jepa-tokenized-triples-v1",
        "created_at_unix": time.time(),
        "config": asdict(cfg),
        "tokenizer_path": cfg.tokenizer_path,
        "vocab_size": len(tokenizer),
        "pad_token_id": pad_id,
        "max_len": cfg.max_len,
        "dtype": "uint16",
        "tokens_shape": ["examples", 3, cfg.max_len],
        "view_axis": {"anchor": 0, "positive": 1, "negative": 2},
        "shards": [],
        "counts": Counter(),
        "transform_vocab": transform_vocab,
    }

    buffer: list[dict[str, str]] = []
    shard_index = 0
    total_examples = 0
    roots = [Path(root).expanduser().resolve() for root in cfg.input_roots]
    pairs = [(root, pair) for root in roots for pair in discover_pairs(root, cfg.max_shards_per_root)]
    if not pairs:
        raise FileNotFoundError(f"no matched views/triples shards under {roots}")

    progress = tqdm(total=cfg.max_examples, desc="tokenize-triples", unit="triple")
    for root, (views_path, triples_path) in pairs:
        examples = examples_from_pair(root, views_path, triples_path, manifest)
        for example in examples:
            buffer.append(example)
            total_examples += 1
            progress.update(1)
            if len(buffer) >= cfg.output_shard_size:
                shard_index = flush_shard(out, shard_index, buffer, tokenizer, cfg, transform_vocab, manifest)
                buffer.clear()
            if cfg.max_examples is not None and total_examples >= cfg.max_examples:
                break
        if cfg.max_examples is not None and total_examples >= cfg.max_examples:
            break

    if buffer:
        flush_shard(out, shard_index, buffer, tokenizer, cfg, transform_vocab, manifest)
        buffer.clear()
    progress.close()

    manifest["counts"] = dict(manifest["counts"])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "done", "output_dir": str(out), "examples": total_examples, "shards": len(manifest["shards"])}), flush=True)
    if cfg.s3_output_prefix:
        sync_s3(out, cfg.s3_output_prefix)


def discover_pairs(root: Path, max_shards: int | None) -> list[tuple[Path, Path]]:
    triples_dir = root / "triples"
    views_dir = root / "views"
    pairs: list[tuple[Path, Path]] = []
    for triples_path in sorted(triples_dir.rglob("*.parquet")):
        rel = triples_path.relative_to(triples_dir)
        views_path = views_dir / rel
        if views_path.exists():
            pairs.append((views_path, triples_path))
    if max_shards is not None:
        pairs = pairs[:max_shards]
    return pairs


def examples_from_pair(
    root: Path,
    views_path: Path,
    triples_path: Path,
    manifest: dict[str, Any],
) -> list[dict[str, str]]:
    view_table = pq.read_table(views_path, columns=["view_id", "code"])
    view_by_id = {
        view_id: code
        for view_id, code in zip(view_table.column("view_id").to_pylist(), view_table.column("code").to_pylist())
        if view_id and code
    }
    triple_table = pq.read_table(
        triples_path,
        columns=[
            "anchor_view_id",
            "positive_view_id",
            "negative_view_id",
            "positive_transform",
            "negative_transform",
            "negative_type",
        ],
    )
    examples: list[dict[str, str]] = []
    for anchor_id, positive_id, negative_id, positive_transform, negative_transform, negative_type in zip(
        triple_table.column("anchor_view_id").to_pylist(),
        triple_table.column("positive_view_id").to_pylist(),
        triple_table.column("negative_view_id").to_pylist(),
        triple_table.column("positive_transform").to_pylist(),
        triple_table.column("negative_transform").to_pylist(),
        triple_table.column("negative_type").to_pylist(),
    ):
        anchor = view_by_id.get(anchor_id)
        positive = view_by_id.get(positive_id)
        negative = view_by_id.get(negative_id)
        if not anchor or not positive or not negative:
            manifest["counts"]["skipped_missing_view"] += 1
            continue
        examples.append(
            {
                "anchor": anchor,
                "positive": positive,
                "negative": negative,
                "positive_transform": positive_transform or "unknown",
                "negative_transform": negative_transform or "unknown",
                "negative_type": negative_type or "unknown",
                "source_root": root.name,
                "source_shard": triples_path.name,
            }
        )
    manifest["counts"]["input_triples"] += triple_table.num_rows
    manifest["counts"]["usable_triples"] += len(examples)
    return examples


def flush_shard(
    out: Path,
    shard_index: int,
    examples: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerFast,
    cfg: Config,
    transform_vocab: dict[str, dict[str, int]],
    manifest: dict[str, Any],
) -> int:
    tokens = np.empty((len(examples), 3, cfg.max_len), dtype=np.uint16)
    positive_transform_id = np.empty((len(examples),), dtype=np.uint16)
    negative_transform_id = np.empty((len(examples),), dtype=np.uint16)
    negative_type_id = np.empty((len(examples),), dtype=np.uint16)

    for start in range(0, len(examples), cfg.tokenize_batch_size):
        batch_examples = examples[start : start + cfg.tokenize_batch_size]
        texts = [item["anchor"] for item in batch_examples]
        texts += [item["positive"] for item in batch_examples]
        texts += [item["negative"] for item in batch_examples]
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=cfg.max_len,
            return_attention_mask=False,
        )["input_ids"]
        arr = np.asarray(encoded, dtype=np.uint16)
        n = len(batch_examples)
        tokens[start : start + n, 0, :] = arr[:n]
        tokens[start : start + n, 1, :] = arr[n : 2 * n]
        tokens[start : start + n, 2, :] = arr[2 * n :]

    for idx, example in enumerate(examples):
        positive_transform_id[idx] = vocab_id(transform_vocab["positive"], example["positive_transform"])
        negative_transform_id[idx] = vocab_id(transform_vocab["negative"], example["negative_transform"])
        negative_type_id[idx] = vocab_id(transform_vocab["negative_type"], example["negative_type"])

    shard_name = f"shard-{shard_index:06d}.npz"
    path = out / shard_name
    tmp = out / f"{shard_name}.tmp"
    np.savez(
        tmp,
        tokens=tokens,
        positive_transform_id=positive_transform_id,
        negative_transform_id=negative_transform_id,
        negative_type_id=negative_type_id,
    )
    tmp.with_suffix(tmp.suffix + ".npz").replace(path)
    manifest["shards"].append(
        {
            "path": shard_name,
            "examples": len(examples),
            "bytes": path.stat().st_size,
        }
    )
    manifest["counts"]["written_examples"] += len(examples)
    return shard_index + 1


def vocab_id(vocab: dict[str, int], value: str) -> int:
    if value not in vocab:
        vocab[value] = len(vocab)
    return vocab[value]


def sync_s3(out: Path, prefix: str) -> None:
    subprocess.run(
        ["s5cmd", "cp", f"{out}/*", f"{prefix.rstrip('/')}/"],
        check=True,
    )


if __name__ == "__main__":
    main()
