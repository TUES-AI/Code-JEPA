#!/usr/bin/env python3
"""Profile the main Siamese BPE Code-JEPA training step by segment."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
import numpy as np

from code_jepa.training.siamese_bpe_jepa import (
    SiameseModel,
    TokenizedShardLoader,
    TrainConfig,
    create_state,
    eval_step,
    train_step,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dirs", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--pad-token-id", type=int, default=0)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--projection-dim", type=int, default=512)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--intermediate-size", type=int, default=2048)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--profile-steps", type=int, default=50)
    p.add_argument("--loader-prefetch", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--forward-profile-steps", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(
        data_dirs=args.data_dirs,
        output_dir=str(out),
        vocab_size=args.vocab_size,
        pad_token_id=args.pad_token_id,
        max_len=args.max_len,
        batch_size=args.batch_size,
        seed=args.seed,
        hidden_size=args.hidden_size,
        projection_dim=args.projection_dim,
        layers=args.layers,
        heads=args.heads,
        intermediate_size=args.intermediate_size,
        precision=args.precision,
        loader_prefetch=args.loader_prefetch,
    )
    (out / "profile-config.json").write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True) + "\n")

    t0 = time.perf_counter()
    loader = TokenizedShardLoader([Path(d) for d in args.data_dirs], batch_size=args.batch_size, seed=args.seed, prefetch=args.loader_prefetch)
    loader_init_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    state = create_state(SiameseModel(cfg), cfg)
    jax.block_until_ready(jax.tree.leaves(state.params)[0])
    model_init_s = time.perf_counter() - t0

    rng = jax.random.PRNGKey(args.seed)
    records: list[dict[str, float | int | str]] = []

    def timed_step(step: int, phase: str):
        nonlocal state, rng
        t_step = time.perf_counter()
        t0 = time.perf_counter()
        batch = loader.next_batch()
        if batch.shape[-1] != args.max_len:
            batch = batch[:, :, : args.max_len]
        loader_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        dev_batch = jax.device_put(batch)
        jax.block_until_ready(dev_batch)
        device_put_s = time.perf_counter() - t0

        rng, step_rng = jax.random.split(rng)
        t0 = time.perf_counter()
        state, metrics = train_step(
            state,
            dev_batch,
            step_rng,
            cfg.margin,
            cfg.pos_weight,
            cfg.rank_weight,
            cfg.inbatch_weight,
            cfg.temperature,
            cfg.sigreg_weight,
            cfg.sigreg_slices,
        )
        jax.block_until_ready(metrics["loss"])
        train_s = time.perf_counter() - t0
        total_s = time.perf_counter() - t_step
        record = {
            "phase": phase,
            "step": step,
            "loader_s": loader_s,
            "device_put_s": device_put_s,
            "train_s": train_s,
            "total_s": total_s,
            "examples_per_s_total": args.batch_size / total_s,
            "examples_per_s_train": args.batch_size / train_s,
            "tokens_per_s_total": args.batch_size * 3 * args.max_len / total_s,
            "loss": float(metrics["loss"]),
            "rank_acc": float(metrics["rank_acc"]),
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        records.append(record)

    timed_step(1, "compile")
    for i in range(2, args.warmup_steps + 2):
        timed_step(i, "warmup")
    for i in range(args.warmup_steps + 2, args.warmup_steps + args.profile_steps + 2):
        timed_step(i, "profile")

    forward_records = []
    for i in range(args.forward_profile_steps + 1):
        t0 = time.perf_counter()
        batch = loader.next_batch()
        if batch.shape[-1] != args.max_len:
            batch = batch[:, :, : args.max_len]
        dev_batch = jax.device_put(batch)
        jax.block_until_ready(dev_batch)
        put_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        metrics = eval_step(state, dev_batch)
        jax.block_until_ready(metrics["eval_rank_acc"])
        forward_s = time.perf_counter() - t0
        phase = "forward_compile" if i == 0 else "forward_profile"
        record = {
            "phase": phase,
            "step": i,
            "device_put_s": put_s,
            "forward_s": forward_s,
            "examples_per_s_forward": args.batch_size / forward_s,
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        forward_records.append(record)

    profile = [r for r in records if r["phase"] == "profile"]
    forward_profile = [r for r in forward_records if r["phase"] == "forward_profile"]
    summary = {
        "loader_init_s": loader_init_s,
        "model_init_s": model_init_s,
        "profile_steps": len(profile),
    }
    for key in ["loader_s", "device_put_s", "train_s", "total_s", "examples_per_s_total", "examples_per_s_train", "tokens_per_s_total"]:
        vals = [float(r[key]) for r in profile]
        summary[f"mean_{key}"] = statistics.mean(vals)
        summary[f"median_{key}"] = statistics.median(vals)
    if forward_profile:
        forward_vals = [float(r["forward_s"]) for r in forward_profile]
        summary["mean_forward_s"] = statistics.mean(forward_vals)
        summary["median_forward_s"] = statistics.median(forward_vals)
        summary["train_to_forward_ratio"] = summary["median_train_s"] / summary["median_forward_s"]
    (out / "profile-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "summary", **summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
