#!/usr/bin/env python3
"""Tokenize prepared Code-JEPA segments into a recursive bucketed cache."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_SCRIPT = ROOT / "scripts" / "tokenize_jepa_triples.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", required=True, help="Prepared dataset root, e.g. .../segments/codesearchnet")
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--languages", nargs="*", default=["all"])
    p.add_argument("--stages", nargs="*", default=["all"])
    p.add_argument("--subsegments", nargs="*", default=["all"])
    p.add_argument("--bucket-lengths", nargs="*", type=int, default=[128, 256, 512, 1024, 2048])
    p.add_argument("--max-len", type=int, default=None)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--output-shard-size", type=int, default=8192)
    p.add_argument("--tokenize-batch-size", type=int, default=4096)
    p.add_argument("--max-examples-per-segment", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    max_len = int(args.max_len or max(args.bucket_lengths))
    segments = discover_segments(input_root, args)
    if not segments:
        raise FileNotFoundError(f"no matching segments under {input_root}")
    (output_root / "_logs").mkdir(parents=True, exist_ok=True)
    started = time.time()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = [executor.submit(run_segment, segment, input_root, output_root, args, max_len) for segment in segments]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            done = len(results)
            elapsed = time.time() - started
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (len(segments) - done) / rate if rate > 0 else None
            print(
                json.dumps(
                    {
                        "event": "segment_done",
                        "done": done,
                        "total": len(segments),
                        "eta_s": round(eta, 1) if eta is not None else None,
                        **result,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    write_root_manifest(output_root, input_root, args, results, max_len)
    print(json.dumps({"event": "done", "output_root": str(output_root), "segments": len(results)}), flush=True)


def discover_segments(input_root: Path, args: argparse.Namespace) -> list[Path]:
    out = []
    for triples_dir in sorted(input_root.rglob("triples")):
        segment = triples_dir.parent
        if not (segment / "views").is_dir():
            continue
        language, stage, subsegment = segment_parts(segment)
        if matches(language, args.languages) and matches(stage, normalize_stages(args.stages)) and matches(subsegment, args.subsegments):
            out.append(segment)
    return out


def run_segment(segment: Path, input_root: Path, output_root: Path, args: argparse.Namespace, max_len: int) -> dict[str, Any]:
    rel = segment.relative_to(input_root)
    out = output_root / rel
    log_path = output_root / "_logs" / ("__".join(rel.parts) + ".log")
    if out.exists() and not (out / "manifest.json").exists():
        if args.overwrite:
            shutil.rmtree(out)
        else:
            raise FileExistsError(f"partial output without manifest: {out}")
    if (out / "manifest.json").exists() and not args.overwrite:
        manifest = json.loads((out / "manifest.json").read_text())
        return {"status": "skipped", "segment": rel.as_posix(), "examples": manifest.get("counts", {}).get("written_examples", 0)}
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    cmd = [
        sys.executable,
        str(TOKENIZER_SCRIPT),
        "--input-roots",
        str(segment),
        "--tokenizer-path",
        str(Path(args.tokenizer_path).expanduser().resolve()),
        "--output-dir",
        str(out),
        "--max-len",
        str(max_len),
        "--bucket-lengths",
        *[str(length) for length in args.bucket_lengths],
        "--output-shard-size",
        str(args.output_shard_size),
        "--tokenize-batch-size",
        str(args.tokenize_batch_size),
    ]
    if args.max_examples_per_segment is not None:
        cmd.extend(["--max-examples-per-segment", str(args.max_examples_per_segment)])
    started = time.time()
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"segment failed: {rel}; see {log_path}")
    manifest = json.loads((out / "manifest.json").read_text())
    return {
        "status": "tokenized",
        "segment": rel.as_posix(),
        "examples": manifest.get("counts", {}).get("written_examples", 0),
        "elapsed_s": round(time.time() - started, 1),
    }


def write_root_manifest(output_root: Path, input_root: Path, args: argparse.Namespace, results: list[dict[str, Any]], max_len: int) -> None:
    shards = []
    counts: dict[str, int] = {"written_examples": 0, "segments": 0}
    segment_manifests = []
    for manifest_path in sorted(output_root.rglob("manifest.json")):
        if manifest_path == output_root / "manifest.json":
            continue
        manifest = json.loads(manifest_path.read_text())
        segment_dir = manifest_path.parent
        rel_manifest = manifest_path.relative_to(output_root).as_posix()
        segment_manifests.append(rel_manifest)
        counts["segments"] += 1
        for key, value in manifest.get("counts", {}).items():
            if isinstance(value, int):
                counts[key] = counts.get(key, 0) + value
        for item in manifest.get("shards", []):
            shard = dict(item)
            shard["path"] = (segment_dir / item["path"]).relative_to(output_root).as_posix()
            shards.append(shard)
    root_manifest = {
        "format": "code-jepa-tokenized-segment-root-v1",
        "created_at_unix": time.time(),
        "input_root": str(input_root),
        "tokenizer_path": str(Path(args.tokenizer_path).expanduser().resolve()),
        "max_len": max_len,
        "bucket_lengths": args.bucket_lengths,
        "config": vars(args),
        "counts": counts,
        "segment_manifests": segment_manifests,
        "shards": shards,
        "results": sorted(results, key=lambda item: item["segment"]),
    }
    (output_root / "manifest.json").write_text(json.dumps(root_manifest, indent=2, sort_keys=True) + "\n")


def segment_parts(segment: Path) -> tuple[str, str, str]:
    parts = segment.parts
    stage_index = next((i for i, part in enumerate(parts) if part.startswith("transform-v")), -1)
    if stage_index < 0:
        return "unknown", "unknown", segment.name
    language = parts[stage_index - 1] if stage_index > 0 else "unknown"
    stage = parts[stage_index].removeprefix("transform-")
    subsegment = parts[stage_index + 1] if stage_index + 1 < len(parts) else "legacy"
    return language, stage, subsegment


def normalize_stages(stages: list[str]) -> list[str]:
    return [stage.removeprefix("transform-") for stage in stages]


def matches(value: str, patterns: list[str]) -> bool:
    if not patterns or "all" in patterns:
        return True
    return any(fnmatch(value, pattern) for pattern in patterns)


if __name__ == "__main__":
    main()
