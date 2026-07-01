#!/usr/bin/env python3
"""Profile the no-predictor bucketed Siamese JAX trainer across device counts.

This is the scaling harness for RunPod/Discoverer smoke tests. It runs the same
trainer used for real training, parses its JSONL logs, and reports ETA plus
multi-GPU scaling efficiency against the one-device run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_siamese_bpe_jepa_multigpu.py"


DEFAULT_BUCKET_BATCHES = {
    "h200": ["128:1024", "256:1024", "512:512", "1024:128", "2048:32"],
    "h100": ["128:512", "256:512", "512:128", "1024:32", "2048:8"],
    "a40": ["128:256", "256:256", "512:64", "1024:16", "2048:4"],
    "safe": ["128:128", "256:128", "512:32", "1024:8", "2048:2"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dirs", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device-counts", nargs="+", type=int, default=[1, 2, 4])
    p.add_argument("--duration-minutes", type=float, default=5.0)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--model-size", choices=["roberta_20m", "roberta_25m", "roberta_30m", "custom"], default="roberta_25m")
    p.add_argument("--hardware-preset", choices=["h200", "h100", "a40", "safe", "custom"], default="h200")
    p.add_argument("--bucket-batches", nargs="*", default=None, help="Per-device batches, e.g. 128:256 256:256 512:64 1024:16 2048:4")
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--projection-dim", type=int, default=512)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--intermediate-size", type=int, default=2048)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--sigreg-weight", type=float, default=0.05)
    p.add_argument("--inbatch-weight", type=float, default=0.1)
    p.add_argument("--rank-weight", type=float, default=1.0)
    p.add_argument("--pos-weight", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--skip-initial-logs", type=int, default=1)
    p.add_argument("--loader-prefetch", type=int, default=1)
    p.add_argument("--python", default=sys.executable)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    runs = []
    for count in args.device_counts:
        run_dir = output_dir / f"devices-{count}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        command = build_command(args, count, run_dir)
        (run_dir / "profile-command.json").write_text(json.dumps(command, indent=2) + "\n")
        print(json.dumps({"event": "profile_run_start", "devices": count, "output_dir": str(run_dir), "command": command}), flush=True)
        with (run_dir / "profile-subprocess.log").open("w") as log:
            proc = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise RuntimeError(f"profile run failed for {count} devices; see {run_dir / 'profile-subprocess.log'}")
        result = summarize_run(run_dir, count, args.skip_initial_logs)
        runs.append(result)
        print(json.dumps({"event": "profile_run_done", **result}, sort_keys=True), flush=True)
    summary = scaling_summary(args, runs, started, bucket_counts(args.data_dirs))
    (output_dir / "scaling-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "scaling-summary.md").write_text(render_markdown(summary) + "\n")
    print(json.dumps({"event": "scaling_summary", **summary}, sort_keys=True), flush=True)


def build_command(args: argparse.Namespace, device_count: int, run_dir: Path) -> list[str]:
    bucket_batches = args.bucket_batches
    hardware_preset = args.hardware_preset
    if hardware_preset != "custom" and bucket_batches is None:
        bucket_batches = DEFAULT_BUCKET_BATCHES[hardware_preset]
    command = [
        args.python,
        str(TRAIN_SCRIPT),
        "--data-dirs",
        *args.data_dirs,
        "--output-dir",
        str(run_dir),
        "--num-devices",
        str(device_count),
        "--hardware-preset",
        hardware_preset,
        "--model-size",
        args.model_size,
        "--vocab-size",
        str(args.vocab_size),
        "--max-len",
        str(args.max_len),
        "--steps",
        str(args.steps),
        "--duration-minutes",
        str(args.duration_minutes),
        "--log-every",
        str(args.log_every),
        "--eval-every",
        "0",
        "--save-every",
        "0",
        "--s3-sync-every",
        "0",
        "--loader-prefetch",
        str(args.loader_prefetch),
        "--precision",
        args.precision,
        "--lr",
        str(args.lr),
        "--warmup-steps",
        str(args.warmup_steps),
        "--sigreg-weight",
        str(args.sigreg_weight),
        "--inbatch-weight",
        str(args.inbatch_weight),
        "--rank-weight",
        str(args.rank_weight),
        "--pos-weight",
        str(args.pos_weight),
        "--temperature",
        str(args.temperature),
        "--margin",
        str(args.margin),
    ]
    if args.model_size == "custom":
        command.extend(
            [
                "--hidden-size",
                str(args.hidden_size),
                "--projection-dim",
                str(args.projection_dim),
                "--layers",
                str(args.layers),
                "--heads",
                str(args.heads),
                "--intermediate-size",
                str(args.intermediate_size),
            ]
        )
    if bucket_batches:
        command.extend(["--bucket-batches", *bucket_batches])
    return command


def summarize_run(run_dir: Path, device_count: int, skip_initial_logs: int) -> dict[str, Any]:
    records = read_jsonl(run_dir / "metrics.jsonl")
    startup = next((item for item in records if item.get("event") == "startup"), {})
    train = [item for item in records if item.get("event") == "train"]
    if not train:
        raise ValueError(f"no train records in {run_dir / 'metrics.jsonl'}")
    steady = train[skip_initial_logs:] or train
    examples_per_s = [float(item["examples_per_s"]) for item in steady]
    tokens_per_s = [float(item["tokens_per_s"]) for item in steady]
    loader_s = [float(item.get("loader_s", 0.0)) for item in steady]
    step_s = [float(item.get("step_s", item.get("batch_s", 0.0))) for item in steady]
    tokenized_examples = int(startup.get("tokenized_examples", train[-1].get("examples_seen", 0)))
    median_eps = statistics.median(examples_per_s)
    seq_lens = sorted({int(item["seq_len"]) for item in steady})
    by_seq_len = {}
    for seq_len in seq_lens:
        seq_records = [item for item in steady if int(item["seq_len"]) == seq_len]
        by_seq_len[str(seq_len)] = {
            "logs": len(seq_records),
            "median_examples_per_s": statistics.median(float(item["examples_per_s"]) for item in seq_records),
            "median_tokens_per_s": statistics.median(float(item["tokens_per_s"]) for item in seq_records),
            "median_batch_s": statistics.median(float(item["batch_s"]) for item in seq_records),
            "median_loader_s": statistics.median(float(item.get("loader_s", 0.0)) for item in seq_records),
            "median_step_s": statistics.median(float(item.get("step_s", item.get("batch_s", 0.0))) for item in seq_records),
        }
    return {
        "devices": device_count,
        "output_dir": str(run_dir),
        "logs": len(train),
        "steady_logs": len(steady),
        "tokenized_examples": tokenized_examples,
        "median_examples_per_s": median_eps,
        "mean_examples_per_s": statistics.mean(examples_per_s),
        "median_tokens_per_s": statistics.median(tokens_per_s),
        "mean_tokens_per_s": statistics.mean(tokens_per_s),
        "median_loader_s": statistics.median(loader_s),
        "median_step_s": statistics.median(step_s),
        "est_hours_per_epoch": round(tokenized_examples / max(median_eps, 1e-9) / 3600, 3),
        "observed_seq_lens": seq_lens,
        "by_seq_len": by_seq_len,
        "last_train": train[-1],
    }


def scaling_summary(
    args: argparse.Namespace,
    runs: list[dict[str, Any]],
    started: float,
    counts_by_bucket: dict[int, int],
) -> dict[str, Any]:
    runs = sorted(runs, key=lambda item: item["devices"])
    base = runs[0]
    base_per_gpu_eps = float(base["median_examples_per_s"]) / int(base["devices"])
    previous = None
    table = []
    for run in runs:
        devices = int(run["devices"])
        eps = float(run["median_examples_per_s"])
        ideal = base_per_gpu_eps * devices
        item = dict(run)
        item["speedup_vs_base"] = round(eps / max(float(base["median_examples_per_s"]), 1e-9), 3)
        item["scaling_efficiency_percent"] = round(100.0 * eps / max(ideal, 1e-9), 1)
        item.update(weighted_eta(run, counts_by_bucket))
        if previous is None:
            item["incremental_examples_per_s"] = 0.0
            item["incremental_efficiency_percent"] = None
        else:
            added_devices = devices - int(previous["devices"])
            delta = eps - float(previous["median_examples_per_s"])
            item["incremental_examples_per_s"] = round(delta, 2)
            item["incremental_efficiency_percent"] = round(100.0 * delta / max(base_per_gpu_eps * added_devices, 1e-9), 1)
        table.append(item)
        previous = run
    return {
        "format": "code-jepa-multigpu-scaling-profile-v1",
        "created_at_unix": time.time(),
        "elapsed_s": round(time.time() - started, 2),
        "config": vars(args),
        "base_devices": int(base["devices"]),
        "base_median_examples_per_s": float(base["median_examples_per_s"]),
        "base_per_gpu_examples_per_s": base_per_gpu_eps,
        "bucket_counts": counts_by_bucket,
        "runs": table,
    }


def bucket_counts(data_dirs: list[str]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for data_dir in data_dirs:
        for manifest_path in Path(data_dir).expanduser().rglob("manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            manifest_counts = manifest.get("counts", {})
            for key, value in manifest_counts.items():
                if key.startswith("bucket_examples:") and isinstance(value, int):
                    counts[int(key.split(":", 1)[1])] = counts.get(int(key.split(":", 1)[1]), 0) + value
    return dict(sorted(counts.items()))


def weighted_eta(run: dict[str, Any], counts_by_bucket: dict[int, int]) -> dict[str, Any]:
    if not counts_by_bucket:
        return {}
    seconds = 0.0
    missing = []
    bucket_hours = {}
    by_seq_len = run.get("by_seq_len", {})
    for bucket, count in counts_by_bucket.items():
        stats = by_seq_len.get(str(bucket))
        if not stats:
            missing.append(bucket)
            continue
        eps = float(stats["median_examples_per_s"])
        hours = count / max(eps, 1e-9) / 3600.0
        bucket_hours[str(bucket)] = round(hours, 3)
        seconds += count / max(eps, 1e-9)
    if missing or seconds <= 0:
        return {"weighted_missing_buckets": missing, "weighted_bucket_hours": bucket_hours}
    total_examples = sum(counts_by_bucket.values())
    return {
        "weighted_est_hours_per_epoch": round(seconds / 3600.0, 3),
        "weighted_effective_examples_per_s": round(total_examples / seconds, 2),
        "weighted_missing_buckets": [],
        "weighted_bucket_hours": bucket_hours,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Multi-GPU Siamese profile summary",
        "",
        f"Base: {summary['base_devices']} device(s), {summary['base_median_examples_per_s']:.2f} examples/s median.",
        "",
        "| Devices | Median ex/s | Raw ETA h | Weighted ETA h | Speedup | Scaling eff. | Incremental eff. |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in summary["runs"]:
        inc_eff = "-" if run["incremental_efficiency_percent"] is None else f"{run['incremental_efficiency_percent']:.1f}%"
        weighted = run.get("weighted_est_hours_per_epoch")
        weighted_text = "-" if weighted is None else f"{weighted:.3f}"
        missing = run.get("weighted_missing_buckets") or []
        if missing:
            weighted_text = f"missing {missing}"
        lines.append(
            "| {devices} | {eps:.2f} | {eta:.3f} | {weighted} | {speedup:.3f}x | {eff:.1f}% | {inc_eff} |".format(
                devices=run["devices"],
                eps=run["median_examples_per_s"],
                eta=run["est_hours_per_epoch"],
                weighted=weighted_text,
                speedup=run["speedup_vs_base"],
                eff=run["scaling_efficiency_percent"],
                inc_eff=inc_eff,
            )
        )
    lines.extend(
        [
            "",
            "Weighted ETA uses root-manifest bucket counts and per-bucket medians when all buckets were sampled.",
            "Per-bucket medians are in `scaling-summary.json` under `runs[].by_seq_len`.",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
