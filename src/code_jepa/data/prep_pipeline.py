"""Tokenizer-agnostic Code-JEPA data-preparation pipeline.

The pipeline writes reproducible Parquet segments by dataset and transform stage. Transform
stage names (`v0`, `v1`, `v2`) refer only to transformation families; the data-prep pipeline
itself is the single canonical pipeline.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import textwrap
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from itertools import combinations, islice
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm

from code_jepa.data.ids import stable_hash
from code_jepa.data.python_ast import ParseResult, ast_spans, line_count, loc_bucket, parse_and_compile, rough_token_len
from code_jepa.transforms.python_ast import (
    TRANSFORM_STAGES,
    TransformResult,
    hard_negative_views_for_stage,
    positive_views_for_stage,
)

CODE_DATASETS = ("codesearchnet_python", "codeparrot_clean_python")
TASK_DATASETS = ("humaneval", "mbpp", "apps", "codecontests")
ALL_DATASETS = CODE_DATASETS + TASK_DATASETS
DEFAULT_TABLES = ("files", "units", "spans", "views", "triples", "relations", "semantic_pairs")


@dataclass(frozen=True)
class PipelineConfig:
    output_dir: str
    datasets: list[str]
    transform_stages: list[str]
    task_datasets: list[str]
    splits: list[str]
    max_examples_per_dataset: int | None = None
    max_files_per_dataset: int | None = None
    dataset_max_files: tuple[str, ...] = ()
    max_units_per_file: int = 96
    max_spans_per_unit: int = 128
    max_positive_views: int = 16
    max_negative_views: int = 8
    min_loc: int = 3
    max_loc: int = 250
    file_window_lines: int = 120
    file_window_stride: int = 80
    max_file_windows: int = 8
    local_window_radius: int = 4
    shard_size: int = 10_000
    compression: str = "zstd"
    num_workers: int = 0
    worker_chunksize: int = 4
    max_task_solutions_per_task: int = 32
    max_semantic_pairs_per_task: int = 256
    worker_buffer_size: int = 256
    streaming: bool = False
    cache_dir: str = ""
    download_only: bool = False
    resume: bool = False
    trust_remote_code: bool = False


@dataclass(frozen=True)
class SourceFile:
    dataset_key: str
    source_dataset: str
    source_split: str
    source_row_index: int
    repository_name: str
    path: str
    language: str
    source: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CodeUnit:
    unit_id: str
    file_id: str
    unit_family: str
    unit_type: str
    qualified_name: str
    code: str
    start_line: int
    end_line: int
    imports_context: str
    class_context: str
    sibling_signatures: str
    identifiers: list[str]
    calls: list[str]
    ast_sequence: list[str]


class SegmentWriter:
    def __init__(
        self,
        root: Path,
        *,
        dataset_key: str,
        segment_name: str,
        cfg: PipelineConfig,
    ) -> None:
        self.root = root
        self.dataset_key = dataset_key
        self.segment_name = segment_name
        self.cfg = cfg
        self.buffers: dict[str, list[dict[str, Any]]] = {table: [] for table in DEFAULT_TABLES}
        self.shard_index: defaultdict[str, int] = defaultdict(int)
        self.counts: Counter[str] = Counter()
        self.root.mkdir(parents=True, exist_ok=True)

    def add(self, table: str, records: list[dict[str, Any]] | dict[str, Any]) -> None:
        if isinstance(records, dict):
            records = [records]
        if not records:
            return
        self.buffers[table].extend(records)
        self.counts[table] += len(records)
        if len(self.buffers[table]) >= self.cfg.shard_size:
            self.flush_table(table)

    def add_many(self, records: dict[str, list[dict[str, Any]]]) -> None:
        for table, rows in records.items():
            self.add(table, rows)

    def flush_table(self, table: str) -> None:
        records = self.buffers[table]
        if not records:
            return
        out = self.root / table / f"shard-{self.shard_index[table]:05d}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}")
        pq.write_table(pa.Table.from_pylist(records), tmp, compression=self.cfg.compression)
        tmp.replace(out)
        self.buffers[table] = []
        self.shard_index[table] += 1

    def close(self) -> None:
        for table in DEFAULT_TABLES:
            self.flush_table(table)
        manifest = {
            "format": "code-jepa-canonical-prep-segment",
            "created_at_unix": time.time(),
            "dataset_key": self.dataset_key,
            "segment_name": self.segment_name,
            "tokenizer_agnostic": True,
            "tables": list(DEFAULT_TABLES),
            "counts": dict(self.counts),
            "config": asdict(self.cfg),
        }
        atomic_write_json(self.root / "manifest.json", manifest)


def parse_args() -> PipelineConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--datasets", nargs="*", default=list(CODE_DATASETS))
    p.add_argument("--transform-stages", nargs="+", default=list(TRANSFORM_STAGES))
    p.add_argument("--task-datasets", nargs="*", default=["humaneval", "mbpp"])
    p.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    p.add_argument("--max-examples-per-dataset", type=int, default=None)
    p.add_argument("--max-files-per-dataset", type=int, default=None)
    p.add_argument("--dataset-max-files", nargs="*", default=())
    p.add_argument("--max-units-per-file", type=int, default=96)
    p.add_argument("--max-spans-per-unit", type=int, default=128)
    p.add_argument("--max-positive-views", type=int, default=16)
    p.add_argument("--max-negative-views", type=int, default=8)
    p.add_argument("--min-loc", type=int, default=3)
    p.add_argument("--max-loc", type=int, default=250)
    p.add_argument("--file-window-lines", type=int, default=120)
    p.add_argument("--file-window-stride", type=int, default=80)
    p.add_argument("--max-file-windows", type=int, default=8)
    p.add_argument("--local-window-radius", type=int, default=4)
    p.add_argument("--shard-size", type=int, default=10_000)
    p.add_argument("--compression", default="zstd")
    p.add_argument("--num-workers", type=int, default=max(0, (os.cpu_count() or 2) - 2))
    p.add_argument("--worker-chunksize", type=int, default=4)
    p.add_argument("--max-task-solutions-per-task", type=int, default=32)
    p.add_argument("--max-semantic-pairs-per-task", type=int, default=256)
    p.add_argument("--worker-buffer-size", type=int, default=256)
    p.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--cache-dir", default="")
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    cfg = PipelineConfig(**vars(p.parse_args()))
    validate_config(cfg)
    return cfg


def validate_config(cfg: PipelineConfig) -> None:
    unknown = sorted(set(cfg.datasets) - set(CODE_DATASETS))
    if unknown:
        raise ValueError(f"unknown code datasets {unknown}; expected subset of {CODE_DATASETS}")
    unknown_tasks = sorted(set(cfg.task_datasets) - set(TASK_DATASETS))
    if unknown_tasks:
        raise ValueError(f"unknown task datasets {unknown_tasks}; expected subset of {TASK_DATASETS}")
    unknown_stages = sorted(set(cfg.transform_stages) - set(TRANSFORM_STAGES))
    if unknown_stages:
        raise ValueError(f"unknown transform stages {unknown_stages}; expected subset of {TRANSFORM_STAGES}")
    for entry in cfg.dataset_max_files:
        if "=" not in entry:
            raise ValueError("--dataset-max-files entries must be dataset=N")
        dataset, value = entry.split("=", 1)
        if dataset not in CODE_DATASETS:
            raise ValueError(f"unknown --dataset-max-files dataset {dataset!r}")
        if int(value) < 1:
            raise ValueError("--dataset-max-files values must be >= 1")
    if cfg.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if cfg.worker_chunksize < 1:
        raise ValueError("--worker-chunksize must be >= 1")
    if cfg.max_task_solutions_per_task < 2:
        raise ValueError("--max-task-solutions-per-task must be >= 2")
    if cfg.max_semantic_pairs_per_task < 1:
        raise ValueError("--max-semantic-pairs-per-task must be >= 1")
    if cfg.worker_buffer_size < 1:
        raise ValueError("--worker-buffer-size must be >= 1")


def main() -> None:
    run_pipeline(parse_args())


def run_pipeline(cfg: PipelineConfig) -> None:
    out = Path(cfg.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if cfg.download_only:
        download_to_hf_cache(cfg)
        return

    pipeline_manifest: dict[str, Any] = {
        "format": "code-jepa-canonical-prep-root",
        "created_at_unix": time.time(),
        "status": "running",
        "tokenizer_agnostic": True,
        "code_datasets": cfg.datasets,
        "task_datasets": cfg.task_datasets,
        "transform_stages": cfg.transform_stages,
        "segments": [],
        "config": asdict(cfg),
    }
    write_pipeline_manifest(out, pipeline_manifest)

    for dataset_key in cfg.datasets:
        for stage in cfg.transform_stages:
            segment_root = out / "segments" / dataset_key / f"transform-{stage}"
            if prepare_segment_root(out, segment_root, cfg.resume):
                pipeline_manifest["segments"].append(str(segment_root.relative_to(out)))
                write_pipeline_manifest(out, pipeline_manifest)
                continue
            writer = SegmentWriter(segment_root, dataset_key=dataset_key, segment_name=f"transform-{stage}", cfg=cfg)
            prepare_code_dataset_segment(dataset_key, stage, cfg, writer)
            writer.close()
            pipeline_manifest["segments"].append(str(segment_root.relative_to(out)))
            write_pipeline_manifest(out, pipeline_manifest)

    for dataset_key in cfg.task_datasets:
        segment_root = out / "segments" / dataset_key / "task-semantic"
        if prepare_segment_root(out, segment_root, cfg.resume):
            pipeline_manifest["segments"].append(str(segment_root.relative_to(out)))
            write_pipeline_manifest(out, pipeline_manifest)
            continue
        writer = SegmentWriter(segment_root, dataset_key=dataset_key, segment_name="task-semantic", cfg=cfg)
        prepare_task_dataset_segment(dataset_key, cfg, writer)
        writer.close()
        pipeline_manifest["segments"].append(str(segment_root.relative_to(out)))
        write_pipeline_manifest(out, pipeline_manifest)

    pipeline_manifest["status"] = "complete"
    write_pipeline_manifest(out, pipeline_manifest)


def write_pipeline_manifest(out: Path, manifest: dict[str, Any]) -> None:
    atomic_write_json(out / "manifest.json", manifest)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def prepare_segment_root(out: Path, segment_root: Path, resume: bool) -> bool:
    """Return True when an existing complete segment should be reused."""

    if resume and (segment_root / "manifest.json").exists():
        return True
    if segment_root.exists():
        segment_root.relative_to(out)
        shutil.rmtree(segment_root)
    return False


def download_to_hf_cache(cfg: PipelineConfig) -> None:
    warm_cfg = PipelineConfig(**{**asdict(cfg), "streaming": False})
    for dataset_key in [*cfg.datasets, *cfg.task_datasets]:
        for args in dataset_load_args(dataset_key, cfg.splits):
            load_dataset_safe(args, warm_cfg)


def prepare_code_dataset_segment(
    dataset_key: str,
    transform_stage: str,
    cfg: PipelineConfig,
    writer: SegmentWriter,
) -> None:
    rows = iter_source_files(dataset_key, cfg)
    total = dataset_file_limit(dataset_key, cfg) or cfg.max_examples_per_dataset
    desc = f"{dataset_key}/{transform_stage}"
    if cfg.num_workers > 1:
        with ProcessPoolExecutor(max_workers=cfg.num_workers) as pool:
            jobs = ((source_file, transform_stage, cfg) for source_file in rows)
            mapped = bounded_process_map(pool, _records_from_source_file_worker, jobs, cfg)
            for records in tqdm(mapped, total=total, desc=desc, unit="file"):
                if records:
                    writer.add_many(records)
        return
    for source_file in tqdm(rows, total=total, desc=desc, unit="file"):
        records = records_from_source_file(source_file, transform_stage, cfg)
        if records:
            writer.add_many(records)


def _records_from_source_file_worker(args: tuple[SourceFile, str, PipelineConfig]) -> dict[str, list[dict[str, Any]]]:
    source_file, transform_stage, cfg = args
    return records_from_source_file(source_file, transform_stage, cfg)


def prepare_task_dataset_segment(dataset_key: str, cfg: PipelineConfig, writer: SegmentWriter) -> None:
    tasks = iter_task_records(dataset_key, cfg)
    if cfg.num_workers > 1:
        with ProcessPoolExecutor(max_workers=cfg.num_workers) as pool:
            jobs = ((task, dataset_key, cfg) for task in tasks)
            mapped = bounded_process_map(pool, _records_from_task_worker, jobs, cfg)
            for records in tqdm(mapped, total=cfg.max_examples_per_dataset, desc=f"{dataset_key}/task", unit="task"):
                writer.add_many(records)
        return
    for task in tqdm(tasks, total=cfg.max_examples_per_dataset, desc=f"{dataset_key}/task", unit="task"):
        writer.add_many(records_from_task(task, dataset_key, cfg))


def _records_from_task_worker(args: tuple[dict[str, Any], str, PipelineConfig]) -> dict[str, list[dict[str, Any]]]:
    task, dataset_key, cfg = args
    return records_from_task(task, dataset_key, cfg)


def bounded_process_map(pool: ProcessPoolExecutor, fn: Any, jobs: Iterable[Any], cfg: PipelineConfig) -> Iterable[Any]:
    try:
        return pool.map(
            fn,
            jobs,
            chunksize=cfg.worker_chunksize,
            buffersize=cfg.worker_buffer_size,
        )
    except TypeError:
        return pool.map(fn, jobs, chunksize=cfg.worker_chunksize)


def dataset_load_args(dataset_key: str, splits: list[str]) -> list[dict[str, Any]]:
    if dataset_key == "codeparrot_clean_python":
        return [{"path": "codeparrot/codeparrot-clean", "split": "train"}]
    if dataset_key == "codesearchnet_python":
        return [
            {"path": "code_search_net", "name": "python", "split": split}
            for split in select_splits(splits, ["train", "validation", "test"])
        ]
    if dataset_key == "humaneval":
        return [{"path": "openai/openai_humaneval", "split": "test"}]
    if dataset_key == "mbpp":
        return [
            {"path": "google-research-datasets/mbpp", "split": split}
            for split in select_splits(splits, ["train", "validation", "test"])
        ]
    if dataset_key == "apps":
        return [
            {
                "path": "json",
                "data_files": {split: f"hf://datasets/codeparrot/apps/{split}.jsonl"},
                "split": split,
            }
            for split in select_splits(splits, ["train", "test"])
        ]
    if dataset_key == "codecontests":
        split_map = {"train": "train", "validation": "valid", "valid": "valid", "test": "test"}
        selected = select_splits(splits, ["train", "validation", "test"])
        return [{"path": "deepmind/code_contests", "split": split_map[split]} for split in selected]
    raise ValueError(dataset_key)


def select_splits(requested: list[str], available: list[str]) -> list[str]:
    if "all" in requested:
        return available
    return [split for split in available if split in requested]


def load_dataset_safe(args: dict[str, Any], cfg: PipelineConfig):
    kwargs = dict(args)
    kwargs["streaming"] = cfg.streaming
    if cfg.cache_dir:
        kwargs["cache_dir"] = cfg.cache_dir
    if cfg.trust_remote_code:
        kwargs["trust_remote_code"] = True
    return load_dataset(**kwargs)


def iter_source_files(dataset_key: str, cfg: PipelineConfig) -> Iterator[SourceFile]:
    produced = 0
    file_limit = dataset_file_limit(dataset_key, cfg)
    for args in dataset_load_args(dataset_key, cfg.splits):
        dataset = load_dataset_safe(args, cfg)
        split_name = str(args["split"])
        iterator: Iterable[dict[str, Any]] = dataset
        row_limit = cfg.max_examples_per_dataset
        if row_limit is not None:
            remaining = row_limit - produced
            if remaining <= 0:
                return
            iterator = islice(iterator, remaining)
        for row_index, row in enumerate(iterator):
            source = source_file_from_row(dataset_key, split_name, row_index, row)
            if source is None:
                continue
            produced += 1
            yield source
            if file_limit is not None and produced >= file_limit:
                return
            if row_limit is not None and produced >= row_limit:
                return


def dataset_file_limit(dataset_key: str, cfg: PipelineConfig) -> int | None:
    for entry in cfg.dataset_max_files:
        name, value = entry.split("=", 1)
        if name == dataset_key:
            return int(value)
    return cfg.max_files_per_dataset


def source_file_from_row(
    dataset_key: str, source_split: str, row_index: int, row: dict[str, Any]
) -> SourceFile | None:
    if dataset_key == "codeparrot_clean_python":
        source = row.get("content") or ""
        if not isinstance(source, str) or not source.strip():
            return None
        repo = str(row.get("repo_name") or "")
        path = str(row.get("path") or f"row-{row_index}.py")
        metadata = {
            "license": row.get("license"),
            "hash": row.get("hash"),
            "autogenerated": row.get("autogenerated"),
            "size": row.get("size"),
        }
        return SourceFile(dataset_key, "codeparrot/codeparrot-clean", source_split, row_index, repo, path, "python", source, metadata)

    if dataset_key == "codesearchnet_python":
        code = row.get("whole_func_string") or row.get("func_code_string") or ""
        if not isinstance(code, str) or not code.strip():
            return None
        repo = str(row.get("repository_name") or "")
        path = str(row.get("func_path_in_repository") or f"row-{row_index}.py")
        func_name = str(row.get("func_name") or f"function_{row_index}")
        source = code.rstrip() + "\n"
        metadata = {
            "func_name": func_name,
            "func_code_url": row.get("func_code_url"),
            "documentation": row.get("func_documentation_string"),
            "note": "CodeSearchNet provides function strings, not whole files.",
        }
        return SourceFile(dataset_key, "code_search_net/python", source_split, row_index, repo, path, "python", source, metadata)

    raise ValueError(dataset_key)


def records_from_source_file(
    source_file: SourceFile,
    transform_stage: str,
    cfg: PipelineConfig,
) -> dict[str, list[dict[str, Any]]]:
    try:
        return _records_from_source_file(source_file, transform_stage, cfg)
    except (RecursionError, MemoryError):
        return empty_records()


def _records_from_source_file(
    source_file: SourceFile,
    transform_stage: str,
    cfg: PipelineConfig,
) -> dict[str, list[dict[str, Any]]]:
    records = empty_records()
    source = source_file.source.rstrip() + "\n"
    if source_file.metadata.get("autogenerated") is True or str(source_file.metadata.get("autogenerated", "")).lower() == "true":
        return records
    parsed = parse_and_compile(source)
    if parsed.tree is None or not parsed.parse_ok:
        return records

    file_hash = str(source_file.metadata.get("hash") or stable_hash([source_file.repository_name, source_file.path, source]))
    file_id = stable_hash([source_file.dataset_key, source_file.repository_name, source_file.path, file_hash])
    repo_split = repo_hash_split(source_file.repository_name or source_file.path or file_id)
    imports_context = imports_from_tree(parsed.tree, source)
    top_defs = top_defs_from_tree(parsed.tree)
    file_record = {
        "file_id": file_id,
        "dataset_key": source_file.dataset_key,
        "source_dataset": source_file.source_dataset,
        "source_split": source_file.source_split,
        "split": repo_split,
        "source_row_index": source_file.source_row_index,
        "repository_name": source_file.repository_name,
        "path": source_file.path,
        "language": source_file.language,
        "source_available": True,
        "source": source,
        "char_len": len(source),
        "loc": line_count(source),
        "rough_token_len": rough_token_len(source),
        "parse_ok": parsed.parse_ok,
        "compile_ok": parsed.compile_ok,
        "imports_context": imports_context,
        "top_defs_json": json.dumps(top_defs, ensure_ascii=False),
        "metadata_json": json.dumps(source_file.metadata, ensure_ascii=False, sort_keys=True),
    }
    records["files"].append(file_record)

    units = extract_python_units(file_id, parsed.tree, source, imports_context, cfg)
    for unit in units:
        unit_loc = line_count(unit.code)
        if unit.unit_family in {"function", "method", "nested_function"} and (unit_loc < cfg.min_loc or unit_loc > cfg.max_loc):
            continue
        unit_parse = parse_and_compile(unit.code) if unit.unit_family != "file_window" else parsed
        unit_record = unit_to_record(unit, source_file, repo_split, transform_stage, unit_parse)
        records["units"].append(unit_record)
        records["relations"].append(
            {
                "relation_id": stable_hash([file_id, "contains", unit.unit_id]),
                "file_id": file_id,
                "src_id": file_id,
                "dst_id": unit.unit_id,
                "relation_type": "file_contains_unit",
                "metadata_json": json.dumps({"unit_family": unit.unit_family}, sort_keys=True),
            }
        )
        unit_spans = ast_spans(unit.unit_id, unit.code)[: cfg.max_spans_per_unit]
        records["spans"].extend({**span, "file_id": file_id, "split": repo_split} for span in unit_spans)
        build_views_and_triples(unit, source_file, repo_split, transform_stage, cfg, records)
    return records


def empty_records() -> dict[str, list[dict[str, Any]]]:
    return {table: [] for table in DEFAULT_TABLES}


def unit_to_record(
    unit: CodeUnit,
    source_file: SourceFile,
    repo_split: str,
    transform_stage: str,
    parsed: Any,
) -> dict[str, Any]:
    return {
        "unit_id": unit.unit_id,
        "file_id": unit.file_id,
        "dataset_key": source_file.dataset_key,
        "source_dataset": source_file.source_dataset,
        "source_split": source_file.source_split,
        "split": repo_split,
        "transform_stage": transform_stage,
        "language": source_file.language,
        "unit_family": unit.unit_family,
        "unit_type": unit.unit_type,
        "qualified_name": unit.qualified_name,
        "repository_name": source_file.repository_name,
        "path": source_file.path,
        "start_line": unit.start_line,
        "end_line": unit.end_line,
        "code": unit.code,
        "imports_context": unit.imports_context,
        "class_context": unit.class_context,
        "sibling_signatures": unit.sibling_signatures,
        "identifiers_json": json.dumps(unit.identifiers, ensure_ascii=False),
        "calls_json": json.dumps(unit.calls, ensure_ascii=False),
        "ast_sequence": " ".join(unit.ast_sequence),
        "loc": line_count(unit.code),
        "char_len": len(unit.code),
        "rough_token_len": rough_token_len(unit.code),
        "size_bucket": loc_bucket(line_count(unit.code)),
        "parse_ok": bool(parsed.parse_ok),
        "compile_ok": bool(parsed.compile_ok),
        "error": parsed.error or "",
    }


def positive_support_views_for_segment(
    code: str,
    stage: str,
    *,
    max_views: int,
) -> tuple[list[TransformResult], set[str]]:
    return support_views_for_segment(
        code,
        stage,
        max_views=max_views,
        getter=positive_views_for_stage,
    )


def negative_support_views_for_segment(
    code: str,
    stage: str,
    *,
    max_views: int,
) -> tuple[list[TransformResult], set[str]]:
    return support_views_for_segment(
        code,
        stage,
        max_views=max_views,
        getter=hard_negative_views_for_stage,
    )


def support_views_for_segment(
    code: str,
    stage: str,
    *,
    max_views: int,
    getter: Any,
) -> tuple[list[TransformResult], set[str]]:
    support: list[TransformResult] = []
    delta_names: set[str] = set()
    seen = {code.strip()}
    for item in stage_chain(stage):
        for view in getter(code, item, max_views=max_views):
            normalized = view.code.strip()
            if normalized in seen:
                continue
            seen.add(normalized)
            support.append(view)
            if item == stage:
                delta_names.add(view.name)
    return support, delta_names


def stage_chain(stage: str) -> list[str]:
    if stage == "v0":
        return ["v0"]
    if stage == "v1":
        return ["v0", "v1"]
    if stage == "v2":
        return ["v0", "v1", "v2"]
    raise ValueError(f"unknown transform stage {stage!r}")


def build_views_and_triples(
    unit: CodeUnit,
    source_file: SourceFile,
    split: str,
    transform_stage: str,
    cfg: PipelineConfig,
    records: dict[str, list[dict[str, Any]]],
) -> None:
    positives, positive_delta_names = positive_support_views_for_segment(
        unit.code,
        transform_stage,
        max_views=cfg.max_positive_views,
    )
    negatives, negative_delta_names = negative_support_views_for_segment(
        unit.code,
        transform_stage,
        max_views=cfg.max_negative_views,
    )
    if not positives or not negatives:
        return

    focal_anchor = make_view(unit, source_file, split, transform_stage, "focal_triplet", "anchor", "anchor", unit.code, "safe", [], {})
    context_anchor_code = contextualize(unit.code, unit)
    context_anchor = make_view(
        unit,
        source_file,
        split,
        transform_stage,
        "context_triplet",
        "anchor",
        "anchor_context",
        context_anchor_code,
        "safe",
        [],
        {},
    )
    ast_view = make_view(
        unit,
        source_file,
        split,
        transform_stage,
        "ast_aux",
        "auxiliary",
        "ast_sequence",
        "# <ast>\n# " + " ".join(unit.ast_sequence) + "\n",
        "safe",
        [],
        {"auxiliary_type": "ast_node_type_sequence"},
    )
    records["views"].extend([focal_anchor, context_anchor, ast_view])
    records["relations"].append(
        {
            "relation_id": stable_hash([unit.unit_id, "ast_aux", ast_view["view_id"]]),
            "file_id": unit.file_id,
            "src_id": unit.unit_id,
            "dst_id": ast_view["view_id"],
            "relation_type": "unit_has_ast_aux_view",
            "metadata_json": "{}",
        }
    )

    pos_views = []
    for transform in positives:
        pos_views.append((transform, make_transform_view(unit, source_file, split, transform_stage, "focal_triplet", transform)))
    neg_views = []
    for transform in negatives:
        neg_views.append((transform, make_transform_view(unit, source_file, split, transform_stage, "focal_triplet", transform)))
    records["views"].extend(view for _, view in pos_views)
    records["views"].extend(view for _, view in neg_views)

    context_pos_views = []
    context_neg_views = []
    for transform in positives:
        context_pos_views.append(
            (
                transform,
                make_transform_view(
                    unit,
                    source_file,
                    split,
                    transform_stage,
                    "context_triplet",
                    replace_code(transform, contextualize(transform.code, unit), "positive_context"),
                ),
            )
        )
    for transform in negatives:
        context_neg_views.append(
            (
                transform,
                make_transform_view(
                    unit,
                    source_file,
                    split,
                    transform_stage,
                    "context_triplet",
                    replace_code(transform, contextualize(transform.code, unit), "negative_context"),
                ),
            )
        )
    records["views"].extend(view for _, view in context_pos_views)
    records["views"].extend(view for _, view in context_neg_views)

    for family, anchor, positive_items, negative_items, weight in [
        ("focal_triplet", focal_anchor, pos_views, neg_views, sampling_weight(unit, "focal_triplet")),
        ("context_triplet", context_anchor, context_pos_views, context_neg_views, sampling_weight(unit, "context_triplet")),
    ]:
        for positive, positive_view in positive_items:
            for negative, negative_view in negative_items:
                if positive.name not in positive_delta_names and negative.name not in negative_delta_names:
                    continue
                records["triples"].append(
                    triple_record(unit, split, transform_stage, family, anchor, positive, positive_view, negative, negative_view, weight)
                )

    build_local_span_triples(
        unit,
        source_file,
        split,
        transform_stage,
        cfg,
        focal_anchor,
        positives[0],
        [negative for negative in negatives if negative.name in negative_delta_names],
        records,
    )


def make_transform_view(
    unit: CodeUnit,
    source_file: SourceFile,
    split: str,
    transform_stage: str,
    family: str,
    transform: TransformResult,
) -> dict[str, Any]:
    return make_view(
        unit,
        source_file,
        split,
        transform_stage,
        family,
        transform.role,
        transform.name,
        transform.code,
        transform.confidence,
        transform.changed_spans,
        transform.metadata,
    )


def replace_code(transform: TransformResult, code: str, name: str) -> TransformResult:
    return TransformResult(
        name=name,
        role=transform.role,
        code=code,
        confidence=transform.confidence,
        changed_spans=transform.changed_spans,
        metadata=transform.metadata,
    )


def make_view(
    unit: CodeUnit,
    source_file: SourceFile,
    split: str,
    transform_stage: str,
    family: str,
    role: str,
    transform_name: str,
    code: str,
    confidence: str,
    changed_spans: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    parsed = parse_and_compile(code)
    view_id = stable_hash([unit.unit_id, family, role, transform_name, code])
    return {
        "view_id": view_id,
        "unit_id": unit.unit_id,
        "file_id": unit.file_id,
        "dataset_key": source_file.dataset_key,
        "source_dataset": source_file.source_dataset,
        "split": split,
        "transform_stage": transform_stage,
        "family": family,
        "role": role,
        "transform_name": transform_name,
        "code": code.rstrip() + "\n",
        "parse_ok": parsed.parse_ok,
        "compile_ok": parsed.compile_ok,
        "confidence": confidence,
        "changed_spans_json": json.dumps(changed_spans, ensure_ascii=False),
        "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    }


def triple_record(
    unit: CodeUnit,
    split: str,
    transform_stage: str,
    family: str,
    anchor_view: dict[str, Any],
    positive: TransformResult,
    positive_view: dict[str, Any],
    negative: TransformResult,
    negative_view: dict[str, Any],
    weight: float,
) -> dict[str, Any]:
    negative_type = str(negative.metadata.get("negative_type", negative.name))
    return {
        "triple_id": stable_hash([unit.unit_id, family, anchor_view["view_id"], positive_view["view_id"], negative_view["view_id"]]),
        "unit_id": unit.unit_id,
        "file_id": unit.file_id,
        "split": split,
        "transform_stage": transform_stage,
        "family": family,
        "anchor_view_id": anchor_view["view_id"],
        "positive_view_id": positive_view["view_id"],
        "negative_view_id": negative_view["view_id"],
        "positive_transform": positive.name,
        "negative_transform": negative.name,
        "negative_type": negative_type,
        "positive_strategy_label": "close",
        "positive_semantic_label": "close",
        "positive_local_label": "low_change",
        "negative_strategy_label": "close",
        "negative_semantic_label": "far",
        "negative_local_label": "changed_span_important",
        "sampling_weight": weight,
    }


def build_local_span_triples(
    unit: CodeUnit,
    source_file: SourceFile,
    split: str,
    transform_stage: str,
    cfg: PipelineConfig,
    focal_anchor: dict[str, Any],
    positive: TransformResult,
    negatives: list[TransformResult],
    records: dict[str, list[dict[str, Any]]],
) -> None:
    del focal_anchor
    for negative in negatives:
        for changed in negative.changed_spans[:2]:
            start_line = int(changed.get("start_line") or 1)
            end_line = int(changed.get("end_line") or start_line)
            anchor_code = line_window(unit.code, start_line, end_line, cfg.local_window_radius)
            positive_code = line_window(positive.code, start_line, end_line, cfg.local_window_radius)
            negative_code = line_window(negative.code, start_line, end_line, cfg.local_window_radius)
            if not anchor_code.strip() or not negative_code.strip():
                continue
            span_unit = CodeUnit(
                unit_id=stable_hash([unit.unit_id, "span_window", negative.name, start_line, end_line]),
                file_id=unit.file_id,
                unit_family="span_window",
                unit_type="span_window",
                qualified_name=f"{unit.qualified_name}:L{start_line}-{end_line}",
                code=anchor_code,
                start_line=start_line,
                end_line=end_line,
                imports_context="",
                class_context="",
                sibling_signatures="",
                identifiers=identifiers_from_code(anchor_code),
                calls=calls_from_code(anchor_code),
                ast_sequence=ast_sequence_from_code(anchor_code),
            )
            span_record = unit_to_record(span_unit, source_file, split, transform_stage, parse_and_compile(anchor_code))
            records["units"].append(span_record)
            anchor_view = make_view(span_unit, source_file, split, transform_stage, "local_span_triplet", "anchor", "span_anchor", anchor_code, "safe", [], {"parent_unit_id": unit.unit_id})
            positive_view = make_view(span_unit, source_file, split, transform_stage, "local_span_triplet", "positive", "span_positive", positive_code, "likely", [], {"parent_unit_id": unit.unit_id})
            negative_view = make_view(span_unit, source_file, split, transform_stage, "local_span_triplet", "negative", negative.name, negative_code, negative.confidence, [changed], negative.metadata | {"parent_unit_id": unit.unit_id})
            records["views"].extend([anchor_view, positive_view, negative_view])
            records["triples"].append(
                triple_record(
                    span_unit,
                    split,
                    transform_stage,
                    "local_span_triplet",
                    anchor_view,
                    TransformResult("span_positive", "positive", positive_code, "likely"),
                    positive_view,
                    negative,
                    negative_view,
                    sampling_weight(span_unit, "local_span_triplet"),
                )
            )


def contextualize(focal_code: str, unit: CodeUnit) -> str:
    sections = []
    if unit.imports_context.strip():
        sections.append("# <imports>\n" + unit.imports_context.strip())
    if unit.class_context.strip():
        sections.append("# <class>\n# " + unit.class_context.strip().replace("\n", "\n# "))
    if unit.sibling_signatures.strip():
        sections.append("# <siblings>\n# " + unit.sibling_signatures.strip().replace("\n", "\n# "))
    sections.append("# <focal>\n" + focal_code.rstrip())
    return "\n\n".join(sections) + "\n"


def sampling_weight(unit: CodeUnit, family: str) -> float:
    bucket = loc_bucket(line_count(unit.code))
    if family == "local_span_triplet":
        return 1.0
    if bucket == "tiny":
        return 0.35
    if bucket in {"short", "medium"}:
        return 1.0
    if bucket == "long":
        return 0.6
    return 0.25


def repo_hash_split(key: str) -> str:
    value = int(stable_hash(["split", key], length=8), 16) % 100
    if value < 90:
        return "train"
    if value < 95:
        return "validation"
    return "test"


def extract_python_units(
    file_id: str,
    tree: ast.AST,
    source: str,
    imports_context: str,
    cfg: PipelineConfig,
) -> list[CodeUnit]:
    units: list[CodeUnit] = []
    class_nodes = [node for node in getattr(tree, "body", []) if isinstance(node, ast.ClassDef)]
    for node in class_nodes:
        item = class_summary_unit(file_id, node, source, imports_context)
        if item is not None:
            units.append(item)
            if len(units) >= cfg.max_units_per_file:
                return units

    extractor = _UnitExtractor(file_id, source, imports_context, cfg.max_units_per_file - len(units))
    extractor.visit(tree)
    units.extend(extractor.units)
    if len(units) < cfg.max_units_per_file:
        units.extend(file_window_units(file_id, source, imports_context, cfg, remaining=cfg.max_units_per_file - len(units)))
    return units[: cfg.max_units_per_file]


class _UnitExtractor(ast.NodeVisitor):
    def __init__(self, file_id: str, source: str, imports_context: str, max_units: int) -> None:
        self.file_id = file_id
        self.source = source
        self.imports_context = imports_context
        self.max_units = max_units
        self.units: list[CodeUnit] = []
        self.class_stack: list[ast.ClassDef] = []
        self.function_stack: list[ast.FunctionDef | ast.AsyncFunctionDef] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_stack.append(node)
        for child in node.body:
            self.visit(child)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._add_function(node)
        self.function_stack.append(node)
        for child in node.body:
            self.visit(child)
        self.function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._add_function(node)
        self.function_stack.append(node)
        for child in node.body:
            self.visit(child)
        self.function_stack.pop()

    def _add_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if len(self.units) >= self.max_units:
            return
        segment = ast.get_source_segment(self.source, node)
        if not segment:
            return
        class_names = [item.name for item in self.class_stack]
        parent_funcs = [item.name for item in self.function_stack]
        qualified_name = ".".join([*class_names, *parent_funcs, node.name])
        if self.function_stack:
            unit_family = "nested_function"
        elif self.class_stack:
            unit_family = "method"
        else:
            unit_family = "function"
        code = textwrap.dedent(segment).rstrip() + "\n"
        unit_id = stable_hash([self.file_id, unit_family, qualified_name, getattr(node, "lineno", 0), getattr(node, "end_lineno", 0), code])
        self.units.append(
            CodeUnit(
                unit_id=unit_id,
                file_id=self.file_id,
                unit_family=unit_family,
                unit_type=unit_family,
                qualified_name=qualified_name or node.name,
                code=code,
                start_line=int(getattr(node, "lineno", 0)),
                end_line=int(getattr(node, "end_lineno", 0)),
                imports_context=self.imports_context,
                class_context=class_signature(self.class_stack[-1]) if self.class_stack else "",
                sibling_signatures=sibling_signatures(self.class_stack[-1] if self.class_stack else None, node),
                identifiers=identifiers_from_tree(node),
                calls=calls_from_tree(node),
                ast_sequence=ast_sequence_from_tree(node),
            )
        )


def class_summary_unit(file_id: str, node: ast.ClassDef, source: str, imports_context: str) -> CodeUnit | None:
    signatures = [function_signature(child) for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))]
    bases = [safe_unparse(base) for base in node.bases]
    suffix = f"({', '.join(bases)})" if bases else ""
    code = f"class {node.name}{suffix}:\n    ..."
    if signatures:
        code += "\n" + "\n".join(f"    {sig}" for sig in signatures)
    code = code.rstrip() + "\n"
    return CodeUnit(
        unit_id=stable_hash([file_id, "class_summary", node.name, getattr(node, "lineno", 0), code]),
        file_id=file_id,
        unit_family="class",
        unit_type="class_summary",
        qualified_name=node.name,
        code=code,
        start_line=int(getattr(node, "lineno", 0)),
        end_line=int(getattr(node, "end_lineno", getattr(node, "lineno", 0))),
        imports_context=imports_context,
        class_context=class_signature(node),
        sibling_signatures="\n".join(signatures),
        identifiers=identifiers_from_tree(node),
        calls=calls_from_tree(node),
        ast_sequence=ast_sequence_from_tree(node),
    )


def file_window_units(
    file_id: str,
    source: str,
    imports_context: str,
    cfg: PipelineConfig,
    *,
    remaining: int,
) -> list[CodeUnit]:
    lines = source.splitlines()
    out = []
    if len(lines) < cfg.file_window_lines:
        return out
    for index, start in enumerate(range(0, len(lines), cfg.file_window_stride)):
        if len(out) >= min(remaining, cfg.max_file_windows):
            break
        end = min(len(lines), start + cfg.file_window_lines)
        window = "\n".join(lines[start:end]).rstrip() + "\n"
        if not window.strip():
            continue
        out.append(
            CodeUnit(
                unit_id=stable_hash([file_id, "file_window", index, start, end, window]),
                file_id=file_id,
                unit_family="file_window",
                unit_type="file_window",
                qualified_name=f"file_window_{index:03d}",
                code=window,
                start_line=start + 1,
                end_line=end,
                imports_context=imports_context,
                class_context="",
                sibling_signatures="",
                identifiers=identifiers_from_code(window),
                calls=calls_from_code(window),
                ast_sequence=ast_sequence_from_code(window),
            )
        )
    return out


def imports_from_tree(tree: ast.AST, source: str) -> str:
    lines = []
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            segment = ast.get_source_segment(source, node)
            if segment:
                lines.append(segment)
        elif not isinstance(node, ast.Expr):
            break
    return "\n".join(lines[:128])


def top_defs_from_tree(tree: ast.AST) -> list[dict[str, str]]:
    out = []
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append({"name": node.name, "node_type": type(node).__name__})
    return out[:512]


def class_signature(node: ast.ClassDef) -> str:
    bases = [safe_unparse(base) for base in node.bases]
    suffix = f"({', '.join(bases)})" if bases else ""
    return f"class {node.name}{suffix}: ..."


def function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    try:
        args = ast.unparse(node.args)
    except Exception:
        args = "..."
    return f"{prefix} {node.name}({args}): ..."


def sibling_signatures(class_node: ast.ClassDef | None, current: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    if class_node is None:
        return ""
    signatures = []
    for child in class_node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name != current.name:
            signatures.append(function_signature(child))
    return "\n".join(signatures[:64])


def identifiers_from_tree(tree: ast.AST) -> list[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return sorted(names)[:512]


def calls_from_tree(tree: ast.AST) -> list[str]:
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = call_name(node.func)
            if name and name not in calls:
                calls.append(name)
    return calls[:512]


def ast_sequence_from_tree(tree: ast.AST) -> list[str]:
    return [type(node).__name__ for node in ast.walk(tree)][:1024]


def identifiers_from_code(code: str) -> list[str]:
    parsed = parse_and_compile(code)
    return identifiers_from_tree(parsed.tree) if parsed.tree is not None else []


def calls_from_code(code: str) -> list[str]:
    parsed = parse_and_compile(code)
    return calls_from_tree(parsed.tree) if parsed.tree is not None else []


def ast_sequence_from_code(code: str) -> list[str]:
    parsed = parse_and_compile(code)
    return ast_sequence_from_tree(parsed.tree) if parsed.tree is not None else []


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "..."


def line_window(code: str, start_line: int, end_line: int, radius: int) -> str:
    lines = code.splitlines()
    lo = max(1, start_line - radius)
    hi = min(len(lines), end_line + radius)
    return "\n".join(lines[lo - 1 : hi]).rstrip() + "\n"


def iter_task_records(dataset_key: str, cfg: PipelineConfig) -> Iterator[dict[str, Any]]:
    produced = 0
    for args in dataset_load_args(dataset_key, cfg.splits):
        dataset = load_dataset_safe(args, cfg)
        iterator: Iterable[dict[str, Any]] = dataset
        for row_index, row in enumerate(iterator):
            task = task_from_row(dataset_key, str(args["split"]), row_index, row)
            if not task:
                continue
            produced += 1
            yield task
            if cfg.max_examples_per_dataset is not None and produced >= cfg.max_examples_per_dataset:
                return


def task_from_row(dataset_key: str, split: str, row_index: int, row: dict[str, Any]) -> dict[str, Any] | None:
    if dataset_key == "humaneval":
        task_id = str(row.get("task_id") or row_index)
        code = (row.get("prompt") or "") + (row.get("canonical_solution") or "")
        solutions = [{"language": "python", "code": code.rstrip() + "\n", "solution_id": "canonical"}]
        return {"task_id": task_id, "split": split, "prompt": row.get("prompt") or "", "solutions": solutions, "metadata": dict(row)}

    if dataset_key == "mbpp":
        task_id = str(row.get("task_id") or row.get("id") or row_index)
        code = str(row.get("code") or "").rstrip() + "\n"
        if not code.strip():
            return None
        return {
            "task_id": task_id,
            "split": split,
            "prompt": row.get("text") or row.get("prompt") or "",
            "solutions": [{"language": "python", "code": code, "solution_id": "canonical"}],
            "metadata": dict(row),
        }

    if dataset_key == "apps":
        task_id = str(row.get("problem_id") or row.get("id") or row_index)
        raw_solutions = parse_json_maybe(row.get("solutions"))
        solutions = []
        if isinstance(raw_solutions, list):
            for index, code in enumerate(raw_solutions):
                if isinstance(code, str) and code.strip():
                    solutions.append({"language": "python", "code": code.rstrip() + "\n", "solution_id": str(index)})
        return {"task_id": task_id, "split": split, "prompt": row.get("question") or "", "solutions": solutions, "metadata": scrub_metadata(row)} if solutions else None

    if dataset_key == "codecontests":
        task_id = str(row.get("name") or row.get("description") or row_index)
        solutions = codecontests_solutions(row.get("solutions"))
        return {"task_id": task_id, "split": split, "prompt": row.get("description") or "", "solutions": solutions, "metadata": scrub_metadata(row)} if solutions else None

    raise ValueError(dataset_key)


def records_from_task(task: dict[str, Any], dataset_key: str, cfg: PipelineConfig) -> dict[str, list[dict[str, Any]]]:
    try:
        return _records_from_task(task, dataset_key, cfg)
    except (RecursionError, MemoryError):
        return empty_records()


def _records_from_task(task: dict[str, Any], dataset_key: str, cfg: PipelineConfig) -> dict[str, list[dict[str, Any]]]:
    records = empty_records()
    task_id = str(task["task_id"])
    split = repo_hash_split(f"{dataset_key}:{task_id}")
    file_id = stable_hash([dataset_key, task_id, "task"])
    prompt = str(task.get("prompt") or "")
    records["files"].append(
        {
            "file_id": file_id,
            "dataset_key": dataset_key,
            "source_dataset": dataset_key,
            "source_split": task.get("split") or "",
            "split": split,
            "source_row_index": -1,
            "repository_name": dataset_key,
            "path": f"{task_id}.task",
            "language": "multi" if len({normalize_language(s.get("language")) for s in task["solutions"]}) > 1 else normalize_language(task["solutions"][0].get("language", "python")),
            "source_available": False,
            "source": "",
            "char_len": 0,
            "loc": 0,
            "rough_token_len": 0,
            "parse_ok": True,
            "compile_ok": True,
            "imports_context": "",
            "top_defs_json": "[]",
            "metadata_json": json.dumps(scrub_metadata(task.get("metadata", {})) | {"task_id": task_id, "prompt": prompt}, ensure_ascii=False, sort_keys=True),
        }
    )
    solution_views = []
    for solution in unique_solutions(task["solutions"], max_solutions=cfg.max_task_solutions_per_task):
        code = str(solution["code"]).rstrip() + "\n"
        language = normalize_language(solution.get("language", "python"))
        parsed = parse_code_for_language(code, language)
        unit_id = stable_hash([dataset_key, task_id, solution.get("solution_id"), language, code])
        unit = CodeUnit(
            unit_id=unit_id,
            file_id=file_id,
            unit_family="task_solution",
            unit_type="task_solution",
            qualified_name=f"{task_id}:{solution.get('solution_id')}",
            code=code,
            start_line=1,
            end_line=line_count(code),
            imports_context="",
            class_context="",
            sibling_signatures="",
            identifiers=identifiers_from_tree(parsed.tree) if parsed.tree is not None else [],
            calls=calls_from_tree(parsed.tree) if parsed.tree is not None else [],
            ast_sequence=ast_sequence_from_tree(parsed.tree) if parsed.tree is not None else [],
        )
        source_file = SourceFile(dataset_key, dataset_key, str(task.get("split") or ""), -1, dataset_key, f"{task_id}.task", language, code, {"task_id": task_id})
        records["units"].append(unit_to_record(unit, source_file, split, "task-semantic", parsed))
        view = make_view(unit, source_file, split, "task-semantic", "task_semantic_pair", "solution", "accepted_solution", code, "accepted", [], {"task_id": task_id, "solution_id": solution.get("solution_id")})
        records["views"].append(view)
        solution_views.append(({**solution, "language": language}, unit, view))
    pair_count = 0
    for (left_solution, left_unit, left_view), (right_solution, right_unit, right_view) in combinations(solution_views, 2):
        if pair_count >= cfg.max_semantic_pairs_per_task:
            break
        relation_kind = "cross_language_same_task" if left_solution.get("language") != right_solution.get("language") else "same_task_different_solution"
        records["semantic_pairs"].append(
            {
                "pair_id": stable_hash([dataset_key, task_id, left_view["view_id"], right_view["view_id"]]),
                "task_id": task_id,
                "dataset_key": dataset_key,
                "split": split,
                "family": "task_semantic_pair",
                "relation_type": relation_kind,
                "left_unit_id": left_unit.unit_id,
                "right_unit_id": right_unit.unit_id,
                "left_view_id": left_view["view_id"],
                "right_view_id": right_view["view_id"],
                "left_language": left_solution.get("language", ""),
                "right_language": right_solution.get("language", ""),
                "semantic_label": "close",
                "strategy_label": "far/maybe",
                "local_label": "diffuse",
                "sampling_weight": 1.0,
                "metadata_json": json.dumps({"task_id": task_id, "prompt": prompt[:1000]}, ensure_ascii=False, sort_keys=True),
            }
        )
        pair_count += 1
    return records


def unique_solutions(solutions: list[dict[str, Any]], *, max_solutions: int) -> list[dict[str, Any]]:
    seen = set()
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in solutions:
        code = str(item.get("code") or "").strip()
        language = normalize_language(item.get("language") or "python")
        if not code:
            continue
        key = (language, code)
        if key in seen:
            continue
        seen.add(key)
        by_language[language].append({**item, "language": language, "code": code + "\n"})

    out: list[dict[str, Any]] = []
    languages = sorted(by_language)
    while len(out) < max_solutions and any(by_language.values()):
        for language in languages:
            if by_language[language]:
                out.append(by_language[language].pop(0))
                if len(out) >= max_solutions:
                    break
    return out


def normalize_language(value: Any) -> str:
    text = str(value or "unknown").strip().lower().replace("++", "pp")
    aliases = {
        "py": "python",
        "python3": "python",
        "python 3": "python",
        "python2": "python2",
        "python 2": "python2",
        "c++": "cpp",
        "cpp": "cpp",
        "cxx": "cpp",
    }
    return aliases.get(text, text)


def parse_code_for_language(code: str, language: str) -> ParseResult:
    if language == "python":
        return parse_and_compile(code)
    return ParseResult(None, False, False, f"parse_skipped_non_python:{language}")


def parse_json_maybe(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def scrub_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    out = {}
    for key, item in value.items():
        if key in {"solutions", "incorrect_solutions"}:
            continue
        try:
            json.dumps(item)
            out[key] = item
        except TypeError:
            out[key] = str(item)
    return out


def codecontests_solutions(value: Any) -> list[dict[str, str]]:
    value = parse_json_maybe(value)
    if not isinstance(value, dict):
        return []
    languages = value.get("language") or value.get("languages") or []
    solutions = value.get("solution") or value.get("solutions") or []
    out = []
    for index, code in enumerate(solutions):
        if not isinstance(code, str) or not code.strip():
            continue
        language = language_name(languages[index] if index < len(languages) else "")
        out.append({"language": language, "code": code.rstrip() + "\n", "solution_id": str(index)})
    return out


def language_name(value: Any) -> str:
    if isinstance(value, str):
        text = value.lower()
        class_label_names = {
            "unknown_language": "unknown",
            "python": "python2",
            "cpp": "cpp",
            "python3": "python",
            "java": "java",
        }
        return class_label_names.get(text, normalize_language(text))
    mapping = {
        0: "unknown",
        1: "python2",
        2: "cpp",
        3: "python",
        4: "java",
    }
    return mapping.get(int(value) if str(value).isdigit() else -1, str(value).lower())


if __name__ == "__main__":
    main()
