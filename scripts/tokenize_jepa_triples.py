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
from fnmatch import fnmatch
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
    bucket_lengths: list[int] | None = None
    max_examples: int | None = None
    max_examples_per_segment: int | None = None
    max_shards_per_root: int | None = None
    output_shard_size: int = 8192
    tokenize_batch_size: int = 1024
    seed: int = 0
    languages: list[str] | None = None
    stages: list[str] | None = None
    subsegments: list[str] | None = None
    s3_output_prefix: str = ""


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-roots", nargs="+", required=True)
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--bucket-lengths", nargs="*", type=int, default=None, help="Optional per-shard token lengths, e.g. 128 256 512 1024.")
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--max-examples-per-segment", type=int, default=None)
    p.add_argument("--max-shards-per-root", type=int, default=None)
    p.add_argument("--output-shard-size", type=int, default=8192)
    p.add_argument("--tokenize-batch-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--languages", nargs="*", default=None, help="Language filters, e.g. python or all.")
    p.add_argument("--stages", nargs="*", default=None, help="Stage filters: v0 v1 v2 or transform-v0 names.")
    p.add_argument("--subsegments", nargs="*", default=None, help="Subsegment filters, e.g. core only-foo or all.")
    p.add_argument("--s3-output-prefix", default="")
    return Config(**vars(p.parse_args()))


def main() -> None:
    cfg = parse_args()
    out = Path(cfg.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.tokenizer_path)
    pad_id = int(tokenizer.pad_token_id or 0)
    bucket_lengths = normalized_bucket_lengths(cfg)
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
        "max_len": max(bucket_lengths) if bucket_lengths else cfg.max_len,
        "bucket_lengths": bucket_lengths,
        "dtype": "uint16",
        "tokens_shape": ["examples", 3, cfg.max_len],
        "view_axis": {"anchor": 0, "positive": 1, "negative": 2},
        "shards": [],
        "counts": Counter(),
        "transform_vocab": transform_vocab,
        "segment_filters": {
            "languages": cfg.languages or ["all"],
            "stages": cfg.stages or ["all"],
            "subsegments": cfg.subsegments or ["all"],
        },
    }

    buffer: list[dict[str, str]] = []
    bucket_buffers: dict[int, list[dict[str, str]]] = {length: [] for length in bucket_lengths}
    bucket_shard_indices: dict[int, int] = {length: 0 for length in bucket_lengths}
    shard_index = 0
    total_examples = 0
    roots = [Path(root).expanduser().resolve() for root in cfg.input_roots]
    segments = discover_segments(roots, cfg)
    if not segments:
        raise FileNotFoundError(f"no segment roots with views/triples under {roots}")
    manifest["segments"] = [segment_info(segment) for segment in segments]

    progress = tqdm(total=cfg.max_examples, desc="tokenize-triples", unit="triple")
    for segment in segments:
        segment_examples = 0
        triples = discover_triples(segment, cfg.max_shards_per_root)
        remaining_global = None if cfg.max_examples is None else max(0, cfg.max_examples - total_examples)
        if cfg.max_examples_per_segment is not None or remaining_global is not None:
            limit = cfg.max_examples_per_segment
            if remaining_global is not None:
                limit = remaining_global if limit is None else min(limit, remaining_global)
            segment_iter = examples_from_limited_segment(segment, triples, limit, manifest)
        else:
            unit_view_shards = build_unit_view_shard_index(segment, manifest)
            segment_iter = (
                example
                for triples_path in triples
                for example in examples_from_triples_with_unit_index(segment, triples_path, unit_view_shards, manifest)
            )
        for example in segment_iter:
            total_examples += 1
            segment_examples += 1
            progress.update(1)
            if bucket_lengths:
                bucket_len = bucket_for_example(example, tokenizer, bucket_lengths, manifest)
                bucket_buffers[bucket_len].append(example)
                if len(bucket_buffers[bucket_len]) >= cfg.output_shard_size:
                    bucket_shard_indices[bucket_len] = flush_shard(
                        out,
                        bucket_shard_indices[bucket_len],
                        bucket_buffers[bucket_len],
                        tokenizer,
                        cfg,
                        transform_vocab,
                        manifest,
                        max_len=bucket_len,
                    )
                    bucket_buffers[bucket_len].clear()
            else:
                buffer.append(example)
                if len(buffer) >= cfg.output_shard_size:
                    shard_index = flush_shard(out, shard_index, buffer, tokenizer, cfg, transform_vocab, manifest)
                    buffer.clear()
            if cfg.max_examples is not None and total_examples >= cfg.max_examples:
                break
        manifest["counts"][f"segment_examples:{segment.as_posix()}"] = segment_examples
        if cfg.max_examples is not None and total_examples >= cfg.max_examples:
            break

    if bucket_lengths:
        for bucket_len, bucket_buffer in bucket_buffers.items():
            if bucket_buffer:
                flush_shard(
                    out,
                    bucket_shard_indices[bucket_len],
                    bucket_buffer,
                    tokenizer,
                    cfg,
                    transform_vocab,
                    manifest,
                    max_len=bucket_len,
                )
                bucket_buffer.clear()
    elif buffer:
        flush_shard(out, shard_index, buffer, tokenizer, cfg, transform_vocab, manifest)
        buffer.clear()
    progress.close()

    manifest["counts"] = dict(manifest["counts"])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "done", "output_dir": str(out), "examples": total_examples, "shards": len(manifest["shards"])}), flush=True)
    if cfg.s3_output_prefix:
        sync_s3(out, cfg.s3_output_prefix)


def normalized_bucket_lengths(cfg: Config) -> list[int]:
    if not cfg.bucket_lengths:
        return []
    lengths = sorted(set(int(length) for length in cfg.bucket_lengths))
    if any(length <= 0 for length in lengths):
        raise ValueError(f"bucket lengths must be positive: {cfg.bucket_lengths}")
    if lengths[-1] > cfg.max_len:
        raise ValueError(f"largest bucket length {lengths[-1]} exceeds --max-len {cfg.max_len}")
    return lengths


def bucket_for_example(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizerFast,
    bucket_lengths: list[int],
    manifest: dict[str, Any],
) -> int:
    encoded = tokenizer(
        [example["anchor"], example["positive"], example["negative"]],
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]
    example["_token_ids"] = encoded
    required = max(len(ids) for ids in encoded)
    for length in bucket_lengths:
        if required <= length:
            manifest["counts"][f"bucket_examples:{length}"] += 1
            manifest["counts"][f"bucket_views:{length}"] += 3
            return length
    manifest["counts"][f"bucket_examples:{bucket_lengths[-1]}"] += 1
    manifest["counts"][f"bucket_views:{bucket_lengths[-1]}"] += 3
    manifest["counts"]["truncated_examples_over_largest_bucket"] += 1
    return bucket_lengths[-1]


def discover_segments(roots: list[Path], cfg: Config) -> list[Path]:
    segments: list[Path] = []
    for root in roots:
        if (root / "views").is_dir() and (root / "triples").is_dir():
            candidates = [root]
        else:
            candidates = sorted(path.parent for path in root.rglob("triples") if (path.parent / "views").is_dir())
        for candidate in candidates:
            if candidate not in segments and segment_matches(candidate, cfg):
                segments.append(candidate)
    return segments


def discover_triples(segment: Path, max_shards: int | None) -> list[Path]:
    triples = sorted((segment / "triples").rglob("*.parquet"))
    return triples[:max_shards] if max_shards is not None else triples


def segment_info(segment: Path) -> dict[str, str]:
    language, stage, subsegment = segment_parts(segment)
    return {
        "path": segment.as_posix(),
        "language": language,
        "stage": stage,
        "subsegment": subsegment,
    }


def segment_matches(segment: Path, cfg: Config) -> bool:
    language, stage, subsegment = segment_parts(segment)
    return (
        filter_match(language, cfg.languages)
        and filter_match(stage, normalize_stage_filters(cfg.stages))
        and filter_match(subsegment, cfg.subsegments)
    )


def filter_match(value: str, patterns: list[str] | None) -> bool:
    if not patterns or "all" in patterns:
        return True
    return any(fnmatch(value, pattern) for pattern in patterns)


def normalize_stage_filters(stages: list[str] | None) -> list[str] | None:
    if stages is None:
        return None
    out = []
    for stage in stages:
        out.append(stage.removeprefix("transform-"))
    return out


def segment_parts(segment: Path) -> tuple[str, str, str]:
    parts = segment.parts
    stage_index = next((i for i, part in enumerate(parts) if part.startswith("transform-v")), -1)
    if stage_index < 0:
        return "unknown", "unknown", segment.name
    language = parts[stage_index - 1] if stage_index > 0 else "unknown"
    stage = parts[stage_index].removeprefix("transform-")
    subsegment = parts[stage_index + 1] if stage_index + 1 < len(parts) else "legacy"
    return language, stage, subsegment


def load_all_views(segment: Path) -> dict[str, str]:
    view_by_id: dict[str, str] = {}
    for views_path in sorted((segment / "views").rglob("*.parquet")):
        table = pq.read_table(views_path, columns=["view_id", "code"])
        for view_id, code in zip(table.column("view_id").to_pylist(), table.column("code").to_pylist()):
            if view_id and code:
                view_by_id[view_id] = code
    return view_by_id


def build_unit_view_shard_index(segment: Path, manifest: dict[str, Any]) -> dict[str, list[Path]]:
    unit_to_paths: dict[str, list[Path]] = {}
    entries = 0
    for views_path in sorted((segment / "views").rglob("*.parquet")):
        table = pq.read_table(views_path, columns=["unit_id"])
        for unit_id in set(table.column("unit_id").to_pylist()):
            if not unit_id:
                continue
            paths = unit_to_paths.setdefault(unit_id, [])
            if not paths or paths[-1] != views_path:
                paths.append(views_path)
                entries += 1
    manifest["counts"]["view_unit_index_units"] += len(unit_to_paths)
    manifest["counts"]["view_unit_index_entries"] += entries
    return unit_to_paths


TRIPLE_COLUMNS = [
    "unit_id",
    "anchor_view_id",
    "positive_view_id",
    "negative_view_id",
    "positive_transform",
    "negative_transform",
    "negative_type",
]


def examples_from_limited_segment(
    segment: Path,
    triples_paths: list[Path],
    limit: int | None,
    manifest: dict[str, Any],
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    needed_view_ids: set[str] = set()
    for triples_path in triples_paths:
        table = pq.read_table(triples_path, columns=TRIPLE_COLUMNS)
        manifest["counts"]["input_triples"] += table.num_rows
        for record in records_from_triple_table(table, triples_path):
            records.append(record)
            needed_view_ids.update([record["anchor_view_id"], record["positive_view_id"], record["negative_view_id"]])
            if limit is not None and len(records) >= limit:
                break
        if limit is not None and len(records) >= limit:
            break
    view_by_id = load_needed_views(segment, needed_view_ids)
    examples = examples_from_records(segment, records, view_by_id, manifest)
    return examples


def examples_from_triples(
    segment: Path,
    triples_path: Path,
    view_by_id: dict[str, str],
    manifest: dict[str, Any],
) -> list[dict[str, str]]:
    triple_table = pq.read_table(triples_path, columns=TRIPLE_COLUMNS)
    manifest["counts"]["input_triples"] += triple_table.num_rows
    return examples_from_records(segment, records_from_triple_table(triple_table, triples_path), view_by_id, manifest)


def examples_from_triples_with_unit_index(
    segment: Path,
    triples_path: Path,
    unit_view_shards: dict[str, list[Path]],
    manifest: dict[str, Any],
) -> list[dict[str, str]]:
    triple_table = pq.read_table(triples_path, columns=TRIPLE_COLUMNS)
    manifest["counts"]["input_triples"] += triple_table.num_rows
    records = records_from_triple_table(triple_table, triples_path)
    needed_view_ids: set[str] = set()
    view_paths: set[Path] = set()
    for record in records:
        needed_view_ids.update([record["anchor_view_id"], record["positive_view_id"], record["negative_view_id"]])
        paths = unit_view_shards.get(record["unit_id"], [])
        if not paths:
            manifest["counts"]["missing_unit_view_index"] += 1
        view_paths.update(paths)
    view_by_id = load_needed_views_from_paths(view_paths, needed_view_ids)
    missing = needed_view_ids.difference(view_by_id)
    if missing:
        manifest["counts"]["fallback_view_scans"] += 1
        fallback = load_needed_views(segment, missing)
        view_by_id.update(fallback)
    return examples_from_records(segment, records, view_by_id, manifest)


def records_from_triple_table(table: Any, triples_path: Path) -> list[dict[str, str]]:
    return [
        {
            "unit_id": unit_id,
            "anchor_view_id": anchor_id,
            "positive_view_id": positive_id,
            "negative_view_id": negative_id,
            "positive_transform": positive_transform or "unknown",
            "negative_transform": negative_transform or "unknown",
            "negative_type": negative_type or "unknown",
            "source_shard": triples_path.name,
        }
        for unit_id, anchor_id, positive_id, negative_id, positive_transform, negative_transform, negative_type in zip(
            table.column("unit_id").to_pylist(),
            table.column("anchor_view_id").to_pylist(),
            table.column("positive_view_id").to_pylist(),
            table.column("negative_view_id").to_pylist(),
            table.column("positive_transform").to_pylist(),
            table.column("negative_transform").to_pylist(),
            table.column("negative_type").to_pylist(),
        )
    ]


def load_needed_views(segment: Path, needed_view_ids: set[str]) -> dict[str, str]:
    return load_needed_views_from_paths(sorted((segment / "views").rglob("*.parquet")), needed_view_ids)


def load_needed_views_from_paths(paths: set[Path] | list[Path], needed_view_ids: set[str]) -> dict[str, str]:
    view_by_id: dict[str, str] = {}
    if not needed_view_ids:
        return view_by_id
    for views_path in sorted(paths):
        table = pq.read_table(views_path, columns=["view_id", "code"])
        view_ids = table.column("view_id").to_pylist()
        if not any(view_id in needed_view_ids for view_id in view_ids):
            continue
        for view_id, code in zip(view_ids, table.column("code").to_pylist()):
            if view_id in needed_view_ids and code:
                view_by_id[view_id] = code
        if len(view_by_id) >= len(needed_view_ids):
            break
    return view_by_id


def examples_from_records(
    segment: Path,
    records: list[dict[str, str]],
    view_by_id: dict[str, str],
    manifest: dict[str, Any],
) -> list[dict[str, str]]:
    language, stage, subsegment = segment_parts(segment)
    examples: list[dict[str, str]] = []
    for record in records:
        anchor = view_by_id.get(record["anchor_view_id"])
        positive = view_by_id.get(record["positive_view_id"])
        negative = view_by_id.get(record["negative_view_id"])
        if not anchor or not positive or not negative:
            manifest["counts"]["skipped_missing_view"] += 1
            continue
        examples.append(
            {
                "anchor": anchor,
                "positive": positive,
                "negative": negative,
                "positive_transform": record["positive_transform"],
                "negative_transform": record["negative_transform"],
                "negative_type": record["negative_type"],
                "source_root": segment.as_posix(),
                "source_shard": record["source_shard"],
                "language": language,
                "stage": stage,
                "subsegment": subsegment,
            }
        )
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
    *,
    max_len: int | None = None,
) -> int:
    shard_max_len = int(max_len or cfg.max_len)
    pad_id = int(tokenizer.pad_token_id or 0)
    tokens = np.full((len(examples), 3, shard_max_len), pad_id, dtype=np.uint16)
    positive_transform_id = np.empty((len(examples),), dtype=np.uint16)
    negative_transform_id = np.empty((len(examples),), dtype=np.uint16)
    negative_type_id = np.empty((len(examples),), dtype=np.uint16)

    if examples and "_token_ids" in examples[0]:
        for row, example in enumerate(examples):
            for view_index, ids in enumerate(example["_token_ids"]):
                clipped = ids[:shard_max_len]
                tokens[row, view_index, : len(clipped)] = np.asarray(clipped, dtype=np.uint16)
    else:
        for start in range(0, len(examples), cfg.tokenize_batch_size):
            batch_examples = examples[start : start + cfg.tokenize_batch_size]
            texts = [item["anchor"] for item in batch_examples]
            texts += [item["positive"] for item in batch_examples]
            texts += [item["negative"] for item in batch_examples]
            encoded = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=shard_max_len,
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

    if max_len is None:
        shard_name = f"shard-{shard_index:06d}.npz"
    else:
        shard_name = f"bucket-{shard_max_len:04d}/shard-{shard_index:06d}.npz"
    path = out / shard_name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out / f"{shard_name}.tmp"
    tmp.parent.mkdir(parents=True, exist_ok=True)
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
            "max_len": shard_max_len,
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
