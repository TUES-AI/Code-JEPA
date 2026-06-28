#!/usr/bin/env python3
"""Augment prepared Code-JEPA units with harder synthetic negative triples.

Reads existing `units/*.parquet` shards and writes a training-compatible root with
new `views/` and `triples/` shards. It does not rewrite raw files or v0 data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from code_jepa.data.ids import stable_hash
from code_jepa.data.python_ast import parse_and_compile
from code_jepa.transforms.python_ast import extra_hard_negative_views, positive_views


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument("--max-positive-views", type=int, default=3)
    parser.add_argument("--max-negative-views", type=int, default=8)
    parser.add_argument("--read-batch-size", type=int, default=8192)
    parser.add_argument("--shard-size-triples", type=int, default=200_000)
    parser.add_argument("--compression", default="zstd")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    unit_shards = discover_unit_shards(args.input_roots)
    if not unit_shards:
        raise FileNotFoundError(f"no units shards under {args.input_roots}")

    stats: dict[str, Any] = {
        "started_at": now_iso(),
        "args": vars(args) | {"input_roots": [str(p) for p in args.input_roots], "output_dir": str(out)},
        "counts": Counter(),
        "positive_transforms": Counter(),
        "negative_transforms": Counter(),
    }
    buffers = {"views": [], "triples": []}
    shard_index = 0
    processed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for path in unit_shards:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(columns=["unit_id", "split", "code"], batch_size=args.read_batch_size):
                rows = [
                    {
                        "unit_id": batch.column("unit_id")[i].as_py(),
                        "split": batch.column("split")[i].as_py(),
                        "code": batch.column("code")[i].as_py(),
                        "max_positive_views": args.max_positive_views,
                        "max_negative_views": args.max_negative_views,
                    }
                    for i in range(batch.num_rows)
                ]
                if args.max_units is not None:
                    remaining = args.max_units - processed
                    if remaining <= 0:
                        break
                    rows = rows[:remaining]
                progress = tqdm(pool.map(process_unit, rows, chunksize=32), total=len(rows), desc=path.name, unit="unit")
                for result in progress:
                    processed += 1
                    merge_stats(stats, result["stats"])
                    if result["views"] and result["triples"]:
                        buffers["views"].extend(result["views"])
                        buffers["triples"].extend(result["triples"])
                    if len(buffers["triples"]) >= args.shard_size_triples:
                        flush(out, buffers, shard_index, args.compression)
                        shard_index += 1
                if args.max_units is not None and processed >= args.max_units:
                    break
            if args.max_units is not None and processed >= args.max_units:
                break

    if buffers["triples"]:
        flush(out, buffers, shard_index, args.compression)
    stats["finished_at"] = now_iso()
    write_json(out / "stats" / "augment-stats.json", normalize_stats(stats))
    write_json(out / "manifests" / "dataset-manifest.json", manifest(args, out, stats))


def discover_unit_shards(roots: list[Path]) -> list[Path]:
    shards: list[Path] = []
    for root in roots:
        shards.extend(sorted((root / "units").glob("*.parquet")))
    return shards


def process_unit(row: dict[str, Any]) -> dict[str, Any]:
    stats = {"counts": Counter(), "positive_transforms": Counter(), "negative_transforms": Counter()}
    stats["counts"]["input_units"] += 1
    unit_id = row["unit_id"]
    split = row.get("split") or "train"
    code = (row.get("code") or "").rstrip() + "\n"
    parsed = parse_and_compile(code)
    if not parsed.parse_ok or not parsed.compile_ok:
        stats["counts"]["skipped_parse_or_compile"] += 1
        return {"views": [], "triples": [], "stats": compact(stats)}

    positives = positive_views(code, max_views=int(row["max_positive_views"]))
    negatives = extra_hard_negative_views(code, max_views=int(row["max_negative_views"]))
    if not positives or not negatives:
        stats["counts"]["skipped_no_augmented_pair"] += 1
        return {"views": [], "triples": [], "stats": compact(stats)}

    anchor_view_id = stable_hash([unit_id, "anchor", code])
    views = [view_record(anchor_view_id, unit_id, split, "anchor", "anchor", code, "safe", [], {})]
    pos_ids: list[tuple[str, str]] = []
    neg_ids: list[tuple[str, str, str]] = []

    for transform in positives:
        view_id = stable_hash([unit_id, transform.role, transform.name, transform.code])
        views.append(
            view_record(
                view_id,
                unit_id,
                split,
                transform.role,
                transform.name,
                transform.code,
                transform.confidence,
                transform.changed_spans,
                transform.metadata,
            )
        )
        pos_ids.append((view_id, transform.name))
        stats["positive_transforms"][transform.name] += 1

    for transform in negatives:
        view_id = stable_hash([unit_id, transform.role, transform.name, transform.code])
        neg_type = str(transform.metadata.get("negative_type", transform.name))
        views.append(
            view_record(
                view_id,
                unit_id,
                split,
                transform.role,
                transform.name,
                transform.code,
                transform.confidence,
                transform.changed_spans,
                transform.metadata,
            )
        )
        neg_ids.append((view_id, transform.name, neg_type))
        stats["negative_transforms"][transform.name] += 1

    triples = []
    for positive_view_id, positive_transform in pos_ids:
        for negative_view_id, negative_transform, negative_type in neg_ids:
            triples.append(
                {
                    "triple_id": stable_hash([unit_id, positive_view_id, negative_view_id]),
                    "unit_id": unit_id,
                    "split": split,
                    "anchor_view_id": anchor_view_id,
                    "positive_view_id": positive_view_id,
                    "negative_view_id": negative_view_id,
                    "positive_transform": positive_transform,
                    "negative_transform": negative_transform,
                    "negative_type": negative_type,
                    "positive_strategy_label": "close",
                    "positive_semantic_label": "close",
                    "positive_local_label": "low_change",
                    "negative_strategy_label": "close",
                    "negative_semantic_label": "far",
                    "negative_local_label": "changed_span_important",
                }
            )

    stats["counts"]["kept_units"] += 1
    stats["counts"]["views"] += len(views)
    stats["counts"]["triples"] += len(triples)
    return {"views": views, "triples": triples, "stats": compact(stats)}


def view_record(
    view_id: str,
    unit_id: str,
    split: str,
    role: str,
    transform_name: str,
    code: str,
    confidence: str,
    changed_spans: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    parsed = parse_and_compile(code)
    return {
        "view_id": view_id,
        "unit_id": unit_id,
        "split": split,
        "role": role,
        "transform_name": transform_name,
        "code": code,
        "parse_ok": parsed.parse_ok,
        "compile_ok": parsed.compile_ok,
        "confidence": confidence,
        "changed_spans_json": json.dumps(changed_spans, ensure_ascii=False),
        "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    }


def flush(out: Path, buffers: dict[str, list[dict[str, Any]]], shard_index: int, compression: str) -> None:
    for table, records in buffers.items():
        if not records:
            continue
        path = out / table / f"shard-{shard_index:05d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(records), path, compression=compression)
        buffers[table] = []


def compact(stats: dict[str, Counter]) -> dict[str, dict[str, int]]:
    return {key: dict(value) for key, value in stats.items()}


def merge_stats(dst: dict[str, Any], src: dict[str, dict[str, int]]) -> None:
    for key in ["counts", "positive_transforms", "negative_transforms"]:
        dst[key].update(src.get(key, {}))


def normalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    out = dict(stats)
    out["counts"] = dict(stats["counts"])
    out["positive_transforms"] = dict(stats["positive_transforms"])
    out["negative_transforms"] = dict(stats["negative_transforms"])
    return out


def manifest(args: argparse.Namespace, out: Path, stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "Code-JEPA harder-negative augmentation v1",
        "created_at": now_iso(),
        "output_dir": str(out),
        "source_roots": [str(path) for path in args.input_roots],
        "tables": ["views", "triples"],
        "notes": [
            "Derived from existing prepared units; raw file/unit tables are not rewritten.",
            "Contains new harder synthetic negatives paired with regenerated positives and anchors.",
        ],
        "counts": dict(stats["counts"]),
        "positive_transforms": dict(stats["positive_transforms"]),
        "negative_transforms": dict(stats["negative_transforms"]),
    }


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
