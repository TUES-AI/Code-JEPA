#!/usr/bin/env python3
"""Train a CodeBERT/RoBERTa JEPA ranker over Code-JEPA Parquet triples.

This is the first real backbone trainer: context encoder is a trainable
RoBERTa/CodeBERT model, target encoder is an EMA copy, and positives/negatives
come from the prepared `views` + `triples` shards.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup


@dataclass(frozen=True)
class TrainArgs:
    data_roots: list[str]
    output_dir: str
    model_name: str = "microsoft/codebert-base"
    max_len: int = 256
    batch_size: int = 96
    steps: int = 1_000_000
    duration_hours: float = 10.0
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1_000
    margin: float = 0.2
    rank_weight: float = 0.5
    inbatch_weight: float = 0.1
    temperature: float = 0.05
    ema_decay: float = 0.996
    dropout: float = 0.1
    grad_clip: float = 1.0
    precision: str = "bf16"
    compile: bool = False
    gradient_checkpointing: bool = True
    seed: int = 0
    log_every: int = 20
    eval_every: int = 200
    eval_batches: int = 20
    save_every: int = 1_000
    s3_sync_every: int = 1_000
    s3_output_prefix: str = ""
    max_shards: int | None = None
    dry_run_batches: int = 0
    eval_only: bool = False
    checkpoint: str = ""


@dataclass(frozen=True)
class ShardPair:
    root: str
    name: str
    views_path: str
    triples_path: str


class Predictor(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ShardSampler:
    def __init__(self, pairs: list[ShardPair], *, seed: int) -> None:
        self.pairs = pairs
        self.rng = random.Random(seed)
        self.examples: list[tuple[str, str, str]] = []
        self.index = 0
        self.current: ShardPair | None = None
        self.last_load_s = 0.0

    def next_batch(self, batch_size: int) -> tuple[list[str], list[str], list[str]]:
        anchors: list[str] = []
        positives: list[str] = []
        negatives: list[str] = []
        while len(anchors) < batch_size:
            if self.index >= len(self.examples):
                self._load_random_shard()
            take = min(batch_size - len(anchors), len(self.examples) - self.index)
            chunk = self.examples[self.index : self.index + take]
            self.index += take
            for anchor, positive, negative in chunk:
                anchors.append(anchor)
                positives.append(positive)
                negatives.append(negative)
        return anchors, positives, negatives

    def _load_random_shard(self) -> None:
        started = time.time()
        pair = self.rng.choice(self.pairs)
        view_table = pq.read_table(pair.views_path, columns=["view_id", "code"])
        view_ids = view_table.column("view_id").to_pylist()
        codes = view_table.column("code").to_pylist()
        view_by_id = {view_id: code for view_id, code in zip(view_ids, codes) if view_id and code}

        triple_table = pq.read_table(
            pair.triples_path,
            columns=["anchor_view_id", "positive_view_id", "negative_view_id"],
        )
        anchor_ids = triple_table.column("anchor_view_id").to_pylist()
        positive_ids = triple_table.column("positive_view_id").to_pylist()
        negative_ids = triple_table.column("negative_view_id").to_pylist()

        examples: list[tuple[str, str, str]] = []
        for anchor_id, positive_id, negative_id in zip(anchor_ids, positive_ids, negative_ids):
            anchor = view_by_id.get(anchor_id)
            positive = view_by_id.get(positive_id)
            negative = view_by_id.get(negative_id)
            if anchor and positive and negative:
                examples.append((anchor, positive, negative))
        if not examples:
            raise RuntimeError(f"no usable triples in shard {pair.triples_path}")
        self.rng.shuffle(examples)
        self.examples = examples
        self.index = 0
        self.current = pair
        self.last_load_s = time.time() - started


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-roots", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="microsoft/codebert-base")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--duration-hours", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--rank-weight", type=float, default=0.5)
    parser.add_argument("--inbatch-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--s3-sync-every", type=int, default=1_000)
    parser.add_argument("--s3-output-prefix", default="")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--dry-run-batches", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default="")
    return TrainArgs(**vars(parser.parse_args()))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(args), indent=2, sort_keys=True) + "\n")

    pairs = discover_shards([Path(root) for root in args.data_roots], max_shards=args.max_shards)
    log(out, {"event": "startup", "pairs": len(pairs), "args": asdict(args), "device": device_name()})

    if args.eval_only:
        metrics = evaluate_checkpoint(args, pairs)
        log(out, {"event": "eval_only", **metrics})
        return

    train(args, pairs, out)


def train(args: TrainArgs, pairs: list[ShardPair], out: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    ctx = AutoModel.from_pretrained(args.model_name).to(device)
    target = AutoModel.from_pretrained(args.model_name).to(device)
    target.load_state_dict(ctx.state_dict())
    target.eval().requires_grad_(False)
    hidden_size = int(ctx.config.hidden_size)
    predictor = Predictor(hidden_size, args.dropout).to(device)

    if args.gradient_checkpointing and hasattr(ctx, "gradient_checkpointing_enable"):
        ctx.gradient_checkpointing_enable()
    if args.compile:
        predictor = torch.compile(predictor)  # type: ignore[assignment]

    optimizer = make_optimizer(list(ctx.parameters()) + list(predictor.parameters()), args)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.precision == "fp16"))
    sampler = ShardSampler(pairs, seed=args.seed)
    eval_sampler = ShardSampler(pairs, seed=args.seed + 17)
    started = time.time()
    deadline = started + args.duration_hours * 3600 if args.duration_hours > 0 else math.inf

    for step in range(1, args.steps + 1):
        if time.time() >= deadline:
            log(out, {"event": "deadline", "step": step})
            break
        batch_started = time.time()
        batch = sampler.next_batch(args.batch_size)
        metrics = train_step(
            ctx,
            target,
            predictor,
            tokenizer,
            optimizer,
            scheduler,
            scaler,
            batch,
            args,
            device,
        )
        metrics.update(
            {
                "event": "train",
                "step": step,
                "elapsed_s": round(time.time() - started, 2),
                "batch_s": round(time.time() - batch_started, 3),
                "lr": scheduler.get_last_lr()[0],
                "shard": sampler.current.name if sampler.current else "",
                "shard_load_s": round(sampler.last_load_s, 3),
            }
        )
        if step == 1 or step % args.log_every == 0:
            log(out, metrics)
        if args.dry_run_batches and step >= args.dry_run_batches:
            log(out, {"event": "dry_run_done", "step": step})
            break
        if step % args.eval_every == 0:
            eval_metrics = evaluate_batches(ctx, predictor, tokenizer, eval_sampler, args, device)
            log(out, {"event": "eval", "step": step, **eval_metrics})
        if step % args.save_every == 0:
            save_light_checkpoint(out, ctx, predictor, args, step)
        if args.s3_output_prefix and args.s3_sync_every > 0 and step % args.s3_sync_every == 0:
            sync_s3(out, args.s3_output_prefix)

    save_light_checkpoint(out, ctx, predictor, args, step)
    if args.s3_output_prefix:
        sync_s3(out, args.s3_output_prefix)
    log(out, {"event": "done", "step": step, "elapsed_s": round(time.time() - started, 2)})


def train_step(
    ctx: nn.Module,
    target: nn.Module,
    predictor: nn.Module,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
    batch: tuple[list[str], list[str], list[str]],
    args: TrainArgs,
    device: torch.device,
) -> dict[str, float]:
    ctx.train()
    predictor.train()
    anchors, positives, negatives = batch
    anchor_inputs = tokenize(tokenizer, anchors, args.max_len, device)
    target_inputs = tokenize(tokenizer, positives + negatives, args.max_len, device)
    amp_dtype = autocast_dtype(args.precision)

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda" and amp_dtype is not None)):
        za = encode(ctx, anchor_inputs)
        pred = F.normalize(predictor(za), dim=-1)
        with torch.no_grad():
            z_target = encode(target, target_inputs)
            zp, zn = z_target.chunk(2, dim=0)
        sim_pos = torch.sum(pred * zp, dim=-1)
        sim_neg = torch.sum(pred * zn, dim=-1)
        jepa_loss = F.mse_loss(pred, zp)
        rank_loss = F.relu(args.margin + sim_neg - sim_pos).mean()
        logits = pred @ zp.T / args.temperature
        labels = torch.arange(pred.shape[0], device=pred.device)
        inbatch_loss = F.cross_entropy(logits, labels)
        loss = jepa_loss + args.rank_weight * rank_loss + args.inbatch_weight * inbatch_loss

    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite loss: {float(loss.detach().cpu())}")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        list(ctx.parameters()) + list(predictor.parameters()), args.grad_clip
    )
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    ema_update(target, ctx, args.ema_decay)

    rank_acc = (sim_pos > sim_neg).float().mean()
    return {
        "loss": float(loss.detach().cpu()),
        "jepa_loss": float(jepa_loss.detach().cpu()),
        "rank_loss": float(rank_loss.detach().cpu()),
        "inbatch_loss": float(inbatch_loss.detach().cpu()),
        "sim_pos": float(sim_pos.mean().detach().cpu()),
        "sim_neg": float(sim_neg.mean().detach().cpu()),
        "rank_acc": float(rank_acc.detach().cpu()),
        "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
    }


@torch.no_grad()
def evaluate_batches(
    model: nn.Module,
    predictor: nn.Module,
    tokenizer: Any,
    sampler: ShardSampler,
    args: TrainArgs,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    predictor.eval()
    losses = []
    rank_accs = []
    sim_pos_values = []
    sim_neg_values = []
    amp_dtype = autocast_dtype(args.precision)
    for _ in range(args.eval_batches):
        anchors, positives, negatives = sampler.next_batch(args.batch_size)
        anchor_inputs = tokenize(tokenizer, anchors, args.max_len, device)
        target_inputs = tokenize(tokenizer, positives + negatives, args.max_len, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda" and amp_dtype is not None)):
            pred = F.normalize(predictor(encode(model, anchor_inputs)), dim=-1)
            zp, zn = encode(model, target_inputs).chunk(2, dim=0)
            sim_pos = torch.sum(pred * zp, dim=-1)
            sim_neg = torch.sum(pred * zn, dim=-1)
            losses.append(float(F.relu(args.margin + sim_neg - sim_pos).mean().cpu()))
            rank_accs.append(float((sim_pos > sim_neg).float().mean().cpu()))
            sim_pos_values.append(float(sim_pos.mean().cpu()))
            sim_neg_values.append(float(sim_neg.mean().cpu()))
    return {
        "eval_rank_loss": float(np.mean(losses)),
        "eval_rank_acc": float(np.mean(rank_accs)),
        "eval_sim_pos": float(np.mean(sim_pos_values)),
        "eval_sim_neg": float(np.mean(sim_neg_values)),
    }


def evaluate_checkpoint(args: TrainArgs, pairs: list[ShardPair]) -> dict[str, float]:
    if not args.checkpoint:
        raise ValueError("--eval-only requires --checkpoint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_name = checkpoint.get("model_name", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to(device)
    hidden_size = int(model.config.hidden_size)
    predictor = Predictor(hidden_size, args.dropout).to(device)
    model.load_state_dict(checkpoint["ctx_model"])
    predictor.load_state_dict(checkpoint["predictor"])
    sampler = ShardSampler(pairs, seed=args.seed + 101)
    return evaluate_batches(model, predictor, tokenizer, sampler, args, device)


def encode(model: nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    output = model(**inputs)
    hidden = output.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = torch.sum(hidden * mask, dim=1) / torch.clamp(torch.sum(mask, dim=1), min=1.0)
    return F.normalize(pooled.float(), dim=-1)


def tokenize(tokenizer: Any, texts: list[str], max_len: int, device: torch.device) -> dict[str, torch.Tensor]:
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def ema_update(target: nn.Module, source: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(decay).add_(source_param.data, alpha=1.0 - decay)
        for target_buffer, source_buffer in zip(target.buffers(), source.buffers()):
            target_buffer.copy_(source_buffer)


def save_light_checkpoint(
    out: Path, ctx: nn.Module, predictor: nn.Module, args: TrainArgs, step: int
) -> None:
    payload = {
        "step": step,
        "model_name": args.model_name,
        "ctx_model": {k: v.detach().cpu() for k, v in ctx.state_dict().items()},
        "predictor": {k: v.detach().cpu() for k, v in predictor.state_dict().items()},
        "args": asdict(args),
    }
    tmp = out / "latest.pt.tmp"
    latest = out / "latest.pt"
    torch.save(payload, tmp)
    tmp.replace(latest)
    if step % (args.save_every * 5) == 0:
        shutil.copy2(latest, out / f"checkpoint-step-{step:08d}.pt")


def make_optimizer(params: list[nn.Parameter], args: TrainArgs) -> torch.optim.Optimizer:
    if torch.cuda.is_available():
        try:
            return AdamW(params, lr=args.lr, weight_decay=args.weight_decay, fused=True)
        except TypeError:
            pass
    return AdamW(params, lr=args.lr, weight_decay=args.weight_decay)


def discover_shards(data_roots: list[Path], *, max_shards: int | None) -> list[ShardPair]:
    pairs: list[ShardPair] = []
    for root in data_roots:
        triples_dir = root / "triples"
        views_dir = root / "views"
        if not triples_dir.exists():
            raise FileNotFoundError(f"missing triples dir: {triples_dir}")
        if not views_dir.exists():
            raise FileNotFoundError(f"missing views dir: {views_dir}")
        for triples_path in sorted(triples_dir.glob("*.parquet")):
            views_path = views_dir / triples_path.name
            if views_path.exists():
                pairs.append(
                    ShardPair(
                        root=str(root),
                        name=f"{root.name}/{triples_path.name}",
                        views_path=str(views_path),
                        triples_path=str(triples_path),
                    )
                )
    if not pairs:
        raise FileNotFoundError(f"no matched views/triples shards under {data_roots}")
    if max_shards is not None:
        pairs = pairs[:max_shards]
    return pairs


def autocast_dtype(precision: str) -> torch.dtype | None:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def sync_s3(out: Path, prefix: str) -> None:
    if not prefix:
        return
    try:
        for name in ["latest.pt", "metrics.jsonl", "run.log", "config.json"]:
            path = out / name
            if path.exists():
                subprocess.run(
                    ["s5cmd", "cp", str(path), f"{prefix.rstrip('/')}/{name}"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        subprocess.run(
            ["s5cmd", "sync", "--size-only", f"{out}/*", prefix],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        log(out, {"event": "s3_sync_failed", "error": f"{type(exc).__name__}: {exc}"})


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def log(out: Path, record: dict[str, Any]) -> None:
    record = dict(record)
    record.setdefault("time", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    with (out / "metrics.jsonl").open("a") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
