#!/usr/bin/env python3
"""Prepare CodeSearchNet Python data for Code-JEPA.

Outputs model-agnostic Parquet records:
- files: synthetic file metadata from CodeSearchNet repo/path fields
- units: function/method code units
- spans: AST node spans
- views: anchor, positive, and hard-negative transformed code
- triples: anchor-positive-negative training relations with per-head labels

No tokenization is performed here.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from code_jepa.data.ids import stable_hash
from code_jepa.data.python_ast import ast_spans, line_count, loc_bucket, parse_and_compile, rough_token_len
from code_jepa.transforms.python_ast import hard_negative_views, positive_views


DATASET_NAME = "code_search_net"
DATASET_CONFIG = "python"


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "started_at": now_iso(),
        "dataset": DATASET_NAME,
        "config": DATASET_CONFIG,
        "splits": args.splits,
        "args": vars(args) | {"output_dir": str(output_dir)},
        "counts": Counter(),
        "by_split": defaultdict(Counter),
        "positive_transforms": Counter(),
        "negative_transforms": Counter(),
        "size_buckets": Counter(),
    }

    for split in args.splits:
        process_split(split, args, output_dir, stats)

    stats["finished_at"] = now_iso()
    write_json(output_dir / "stats" / "transform-stats.json", normalize_stats(stats))
    write_json(output_dir / "manifests" / "dataset-manifest.json", manifest(args, output_dir, stats))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Volumes/SSD/datasets/code-jepa/processed/v0/codesearchnet-python"),
    )
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--max-examples-per-split", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=10_000)
    parser.add_argument("--min-loc", type=int, default=3)
    parser.add_argument("--max-loc", type=int, default=250)
    parser.add_argument("--max-positive-views", type=int, default=3)
    parser.add_argument("--max-negative-views", type=int, default=5)
    parser.add_argument("--max-spans-per-unit", type=int, default=256)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compression", default="zstd")
    return parser.parse_args()


def process_split(split: str, args: argparse.Namespace, output_dir: Path, stats: dict[str, Any]) -> None:
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split, streaming=args.streaming)
    iterator: Iterable[dict[str, Any]] = dataset
    if args.max_examples_per_split is not None:
        iterator = take(iterator, args.max_examples_per_split)

    buffers: dict[str, list[dict[str, Any]]] = {
        "files": [],
        "units": [],
        "spans": [],
        "views": [],
        "triples": [],
    }
    seen_files: set[str] = set()
    shard_index = 0

    progress_total = args.max_examples_per_split
    progress = tqdm(iterator, total=progress_total, desc=f"{split}", unit="unit")
    for row_index, row in enumerate(progress):
        stats["counts"]["input_rows"] += 1
        stats["by_split"][split]["input_rows"] += 1
        records = records_from_row(row, row_index, split, args, stats)
        if records is None:
            continue

        file_record = records["file"]
        if file_record["file_id"] not in seen_files:
            seen_files.add(file_record["file_id"])
            buffers["files"].append(file_record)

        for table_name in ("units", "spans", "views", "triples"):
            buffers[table_name].extend(records[table_name])

        if len(buffers["units"]) >= args.shard_size:
            flush_buffers(output_dir, split, shard_index, buffers, args.compression)
            shard_index += 1
            for key in buffers:
                buffers[key].clear()

    flush_buffers(output_dir, split, shard_index, buffers, args.compression)


def records_from_row(
    row: dict[str, Any], row_index: int, split: str, args: argparse.Namespace, stats: dict[str, Any]
) -> dict[str, Any] | None:
    code = row.get("whole_func_string") or row.get("func_code_string") or ""
    code = code.rstrip() + "\n"
    loc = line_count(code)
    if loc < args.min_loc or loc > args.max_loc:
        stats["counts"]["skipped_loc_filter"] += 1
        stats["by_split"][split]["skipped_loc_filter"] += 1
        return None

    parsed = parse_and_compile(code)
    if not parsed.parse_ok or not parsed.compile_ok:
        stats["counts"]["skipped_parse_or_compile"] += 1
        stats["by_split"][split]["skipped_parse_or_compile"] += 1
        return None

    repo = row.get("repository_name") or ""
    path = row.get("func_path_in_repository") or ""
    func_name = row.get("func_name") or ""
    language = row.get("language") or "python"
    file_id = stable_hash([DATASET_NAME, DATASET_CONFIG, repo, path])
    unit_id = stable_hash([DATASET_NAME, DATASET_CONFIG, repo, path, func_name, code])
    anchor_view_id = stable_hash([unit_id, "anchor", code])
    bucket = loc_bucket(loc)

    file_record = {
        "file_id": file_id,
        "dataset": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "split": split,
        "repository_name": repo,
        "path": path,
        "language": language,
        "source_available": False,
        "source": "",
        "metadata_json": json.dumps({"note": "CodeSearchNet provides function strings, not full files."}),
    }

    unit_record = {
        "unit_id": unit_id,
        "file_id": file_id,
        "dataset": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "split": split,
        "source_row_index": row_index,
        "language": language,
        "unit_type": "function",
        "qualified_name": func_name,
        "repository_name": repo,
        "path": path,
        "code_url": row.get("func_code_url") or "",
        "code": code,
        "documentation": row.get("func_documentation_string") or "",
        "loc": loc,
        "char_len": len(code),
        "rough_token_len": rough_token_len(code),
        "size_bucket": bucket,
        "parse_ok": parsed.parse_ok,
        "compile_ok": parsed.compile_ok,
        "error": parsed.error,
    }

    span_records = ast_spans(unit_id, code)[: args.max_spans_per_unit]
    pos = positive_views(code, max_views=args.max_positive_views)
    neg = hard_negative_views(code, max_views=args.max_negative_views)

    view_records = [
        view_record(anchor_view_id, unit_id, split, "anchor", "anchor", code, "safe", [], {})
    ]
    pos_ids: list[tuple[str, str]] = []
    neg_ids: list[tuple[str, str, str]] = []

    for transform in pos:
        view_id = stable_hash([unit_id, transform.role, transform.name, transform.code])
        view_records.append(
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

    for transform in neg:
        view_id = stable_hash([unit_id, transform.role, transform.name, transform.code])
        view_records.append(
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
        neg_type = str(transform.metadata.get("negative_type", transform.name))
        neg_ids.append((view_id, transform.name, neg_type))
        stats["negative_transforms"][transform.name] += 1

    triple_records = []
    for positive_view_id, positive_transform in pos_ids:
        for negative_view_id, negative_transform, negative_type in neg_ids:
            triple_id = stable_hash([unit_id, positive_view_id, negative_view_id])
            triple_records.append(
                {
                    "triple_id": triple_id,
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
    stats["counts"]["views"] += len(view_records)
    stats["counts"]["triples"] += len(triple_records)
    stats["counts"]["spans"] += len(span_records)
    stats["by_split"][split]["kept_units"] += 1
    stats["by_split"][split]["views"] += len(view_records)
    stats["by_split"][split]["triples"] += len(triple_records)
    stats["by_split"][split]["spans"] += len(span_records)
    stats["size_buckets"][bucket] += 1

    return {
        "file": file_record,
        "units": [unit_record],
        "spans": span_records,
        "views": view_records,
        "triples": triple_records,
    }


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


def flush_buffers(
    output_dir: Path,
    split: str,
    shard_index: int,
    buffers: dict[str, list[dict[str, Any]]],
    compression: str,
) -> None:
    for table_name, records in buffers.items():
        if not records:
            continue
        out_path = output_dir / table_name / f"split-{split}" / f"shard-{shard_index:05d}.parquet"
        write_parquet(records, out_path, compression=compression)


def write_parquet(records: list[dict[str, Any]], out_path: Path, *, compression: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, out_path, compression=compression)


def take(items: Iterable[dict[str, Any]], n: int) -> Iterable[dict[str, Any]]:
    for index, item in enumerate(items):
        if index >= n:
            break
        yield item


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def normalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    out = dict(stats)
    out["counts"] = dict(stats["counts"])
    out["by_split"] = {split: dict(counter) for split, counter in stats["by_split"].items()}
    out["positive_transforms"] = dict(stats["positive_transforms"])
    out["negative_transforms"] = dict(stats["negative_transforms"])
    out["size_buckets"] = dict(stats["size_buckets"])
    return out


def manifest(args: argparse.Namespace, output_dir: Path, stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "Code-JEPA CodeSearchNet Python v0",
        "created_at": now_iso(),
        "output_dir": str(output_dir),
        "source_dataset": DATASET_NAME,
        "source_config": DATASET_CONFIG,
        "source_splits": args.splits,
        "tables": ["files", "units", "spans", "views", "triples"],
        "tokenized": False,
        "notes": [
            "Function-level CodeSearchNet source; full file source is not available in this v0 table.",
            "Hard negatives are behavior-impacting mutations relative to the original, not proven failing programs.",
            "No tokenizer-specific fields are written.",
        ],
        "counts": dict(stats["counts"]),
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
