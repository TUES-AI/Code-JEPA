#!/usr/bin/env python3
"""Train a no-predictor Siamese Code-JEPA encoder with a custom BPE tokenizer.

This is a short diagnostic trainer: one shared RoBERTa encoder embeds anchor,
positive, and negative views. There is no predictor, no target encoder, no EMA,
and no stop-grad. SIGReg is applied to encoder outputs.
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
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaModel, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from code_jepa.models import SlicedGaussianRegularizer


@dataclass(frozen=True)
class TrainArgs:
    data_roots: list[str]
    tokenizer_path: str
    output_dir: str
    max_len: int = 256
    batch_size: int = 512
    steps: int = 1_000_000
    duration_minutes: float = 30.0
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    hidden_size: int = 512
    layers: int = 6
    heads: int = 8
    intermediate_size: int = 2048
    margin: float = 0.2
    pos_weight: float = 1.0
    rank_weight: float = 1.0
    inbatch_weight: float = 0.1
    temperature: float = 0.05
    sigreg_weight: float = 0.05
    sigreg_slices: int = 64
    grad_clip: float = 1.0
    precision: str = "bf16"
    seed: int = 0
    log_every: int = 20
    eval_every: int = 200
    eval_batches: int = 20
    save_every: int = 1000
    s3_sync_every: int = 1000
    s3_output_prefix: str = ""
    max_shards: int | None = None
    dry_run_batches: int = 0
    resume: str = ""
    init_checkpoint: str = ""


@dataclass(frozen=True)
class ShardPair:
    root: str
    name: str
    views_path: str
    triples_path: str


class ShardSampler:
    def __init__(self, pairs: list[ShardPair], *, seed: int) -> None:
        self.pairs = pairs
        self.rng = random.Random(seed)
        self.examples: list[tuple[str, str, str, str, str, str]] = []
        self.index = 0
        self.current: ShardPair | None = None
        self.current_pair_index: int | None = None
        self.current_load_rng_state: object | None = None
        self.last_load_s = 0.0

    def next_batch(self, batch_size: int) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
        anchors: list[str] = []
        positives: list[str] = []
        negatives: list[str] = []
        positive_transforms: list[str] = []
        negative_transforms: list[str] = []
        negative_types: list[str] = []
        while len(anchors) < batch_size:
            if self.index >= len(self.examples):
                self._load_random_shard()
            take = min(batch_size - len(anchors), len(self.examples) - self.index)
            chunk = self.examples[self.index : self.index + take]
            self.index += take
            for anchor, positive, negative, positive_transform, negative_transform, negative_type in chunk:
                anchors.append(anchor)
                positives.append(positive)
                negatives.append(negative)
                positive_transforms.append(positive_transform)
                negative_transforms.append(negative_transform)
                negative_types.append(negative_type)
        return anchors, positives, negatives, positive_transforms, negative_transforms, negative_types

    def _load_random_shard(self) -> None:
        started = time.time()
        self.current_load_rng_state = self.rng.getstate()
        pair_index = self.rng.randrange(len(self.pairs))
        pair = self.pairs[pair_index]
        view_table = pq.read_table(pair.views_path, columns=["view_id", "code"])
        view_by_id = {
            view_id: code
            for view_id, code in zip(view_table.column("view_id").to_pylist(), view_table.column("code").to_pylist())
            if view_id and code
        }
        triple_table = pq.read_table(
            pair.triples_path,
            columns=[
                "anchor_view_id",
                "positive_view_id",
                "negative_view_id",
                "positive_transform",
                "negative_transform",
                "negative_type",
            ],
        )
        examples: list[tuple[str, str, str, str, str, str]] = []
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
            if anchor and positive and negative:
                examples.append((anchor, positive, negative, positive_transform or "", negative_transform or "", negative_type or ""))
        if not examples:
            raise RuntimeError(f"no usable triples in shard {pair.triples_path}")
        self.rng.shuffle(examples)
        self.examples = examples
        self.index = 0
        self.current = pair
        self.current_pair_index = pair_index
        self.last_load_s = time.time() - started

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng_state": self.rng.getstate(),
            "index": self.index,
            "current_pair_index": self.current_pair_index,
            "current_name": self.current.name if self.current else "",
            "current_load_rng_state": self.current_load_rng_state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.examples = []
        self.index = 0
        self.current = None
        self.current_pair_index = None
        self.current_load_rng_state = None
        current_load_rng_state = state.get("current_load_rng_state")
        if current_load_rng_state is not None:
            self.rng.setstate(current_load_rng_state)
            self._load_random_shard()
            expected_name = state.get("current_name") or ""
            if expected_name and self.current and self.current.name != expected_name:
                raise RuntimeError(f"sampler state mismatch: expected {expected_name}, got {self.current.name}")
            self.index = min(int(state.get("index", 0)), len(self.examples))
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self.rng.setstate(rng_state)


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-roots", nargs="+", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--duration-minutes", type=float, default=30.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--inbatch-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--sigreg-weight", type=float, default=0.05)
    parser.add_argument("--sigreg-slices", type=int, default=64)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--s3-sync-every", type=int, default=1000)
    parser.add_argument("--s3-output-prefix", default="")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--dry-run-batches", type=int, default=0)
    parser.add_argument("--resume", default="", help="Resume a full training checkpoint, including optimizer/scheduler/scaler/sampler/RNG state.")
    parser.add_argument("--init-checkpoint", default="", help="Initialize model weights from a checkpoint without restoring optimizer or dataloader state.")
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

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
    pairs = discover_shards([Path(root) for root in args.data_roots], max_shards=args.max_shards)
    log(out, {"event": "startup", "pairs": len(pairs), "vocab_size": len(tokenizer), "device": device_name(), "args": asdict(args)})
    train(args, pairs, tokenizer, out)


def train(args: TrainArgs, pairs: list[ShardPair], tokenizer: PreTrainedTokenizerFast, out: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, tokenizer).to(device)
    sigreg = SlicedGaussianRegularizer(num_slices=args.sigreg_slices).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=(device.type == "cuda"))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.precision == "fp16"))
    sampler = ShardSampler(pairs, seed=args.seed)
    eval_sampler = ShardSampler(pairs, seed=args.seed + 17)

    start_step = 0
    if args.resume:
        start_step = load_full_checkpoint(
            resolve_checkpoint_path(args.resume, out),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            sigreg=sigreg,
            sampler=sampler,
            eval_sampler=eval_sampler,
            device=device,
        )
        log(out, {"event": "resumed", "checkpoint": args.resume, "start_step": start_step})
    elif args.init_checkpoint:
        init_step = load_model_checkpoint(resolve_checkpoint_path(args.init_checkpoint, out), model=model, device=device)
        log(out, {"event": "initialized_from_checkpoint", "checkpoint": args.init_checkpoint, "source_step": init_step})

    started = time.time()
    deadline = started + args.duration_minutes * 60 if args.duration_minutes > 0 else math.inf

    step = start_step
    for step in range(start_step + 1, args.steps + 1):
        if time.time() >= deadline:
            log(out, {"event": "deadline", "step": step})
            break
        batch_started = time.time()
        batch = sampler.next_batch(args.batch_size)
        metrics = train_step(model, sigreg, tokenizer, optimizer, scheduler, scaler, batch, args, device)
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
            log(out, {"event": "eval", "step": step, **evaluate_batches(model, tokenizer, eval_sampler, args, device)})
        if step % args.save_every == 0:
            save_checkpoint(out, model, optimizer, scheduler, scaler, sigreg, sampler, eval_sampler, args, step)
        if args.s3_output_prefix and args.s3_sync_every > 0 and step % args.s3_sync_every == 0:
            sync_s3(out, args.s3_output_prefix)

    save_checkpoint(out, model, optimizer, scheduler, scaler, sigreg, sampler, eval_sampler, args, step)
    if args.s3_output_prefix:
        sync_s3(out, args.s3_output_prefix)
    log(out, {"event": "done", "step": step, "elapsed_s": round(time.time() - started, 2)})


def train_step(
    model: nn.Module,
    sigreg: SlicedGaussianRegularizer,
    tokenizer: PreTrainedTokenizerFast,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
    batch: tuple[list[str], list[str], list[str], list[str], list[str], list[str]],
    args: TrainArgs,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    anchors, positives, negatives = batch[:3]
    inputs = tokenize(tokenizer, anchors + positives + negatives, args.max_len, device)
    amp_dtype = autocast_dtype(args.precision)

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda" and amp_dtype is not None)):
        embeddings = encode(model, inputs)
        za, zp, zn = embeddings.chunk(3, dim=0)
        za_n = F.normalize(za, dim=-1)
        zp_n = F.normalize(zp, dim=-1)
        zn_n = F.normalize(zn, dim=-1)
        sim_pos = torch.sum(za_n * zp_n, dim=-1)
        sim_neg = torch.sum(za_n * zn_n, dim=-1)
        pos_loss = 1.0 - sim_pos.mean()
        rank_loss = F.relu(args.margin + sim_neg - sim_pos).mean()
        logits = za_n @ zp_n.T / args.temperature
        labels = torch.arange(za.shape[0], device=device)
        inbatch_loss = F.cross_entropy(logits, labels)
        sigreg_loss = sigreg(torch.cat([za, zp, zn], dim=0))
        loss = (
            args.pos_weight * pos_loss
            + args.rank_weight * rank_loss
            + args.inbatch_weight * inbatch_loss
            + args.sigreg_weight * sigreg_loss
        )

    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite loss: {float(loss.detach().cpu())}")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    return metrics(loss, pos_loss, rank_loss, inbatch_loss, sigreg_loss, sim_pos, sim_neg, grad_norm)


@torch.no_grad()
def evaluate_batches(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerFast,
    sampler: ShardSampler,
    args: TrainArgs,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses = []
    rank_accs = []
    gaps = []
    sim_pos_values = []
    sim_neg_values = []
    by_positive_transform: dict[str, dict[str, float]] = {}
    by_negative_transform: dict[str, dict[str, float]] = {}
    by_negative_type: dict[str, dict[str, float]] = {}
    amp_dtype = autocast_dtype(args.precision)
    for _ in range(args.eval_batches):
        anchors, positives, negatives, positive_transforms, negative_transforms, negative_types = sampler.next_batch(args.batch_size)
        inputs = tokenize(tokenizer, anchors + positives + negatives, args.max_len, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda" and amp_dtype is not None)):
            za, zp, zn = encode(model, inputs).chunk(3, dim=0)
            za = F.normalize(za, dim=-1)
            zp = F.normalize(zp, dim=-1)
            zn = F.normalize(zn, dim=-1)
            sim_pos = torch.sum(za * zp, dim=-1)
            sim_neg = torch.sum(za * zn, dim=-1)
            per_rank_loss = F.relu(args.margin + sim_neg - sim_pos)
            correct = (sim_pos > sim_neg).float()
            gap = sim_pos - sim_neg
            losses.append(float(per_rank_loss.mean().cpu()))
            rank_accs.append(float(correct.mean().cpu()))
            gaps.append(float(gap.mean().cpu()))
            sim_pos_values.append(float(sim_pos.mean().cpu()))
            sim_neg_values.append(float(sim_neg.mean().cpu()))
            update_group_stats(by_positive_transform, positive_transforms, correct, per_rank_loss, gap)
            update_group_stats(by_negative_transform, negative_transforms, correct, per_rank_loss, gap)
            update_group_stats(by_negative_type, negative_types, correct, per_rank_loss, gap)
    return {
        "eval_rank_loss": float(np.mean(losses)),
        "eval_rank_acc": float(np.mean(rank_accs)),
        "eval_sim_gap": float(np.mean(gaps)),
        "eval_sim_pos": float(np.mean(sim_pos_values)),
        "eval_sim_neg": float(np.mean(sim_neg_values)),
        "eval_by_positive_transform": finalize_group_stats(by_positive_transform),
        "eval_by_negative_transform": finalize_group_stats(by_negative_transform),
        "eval_by_negative_type": finalize_group_stats(by_negative_type),
    }


def update_group_stats(
    groups: dict[str, dict[str, float]],
    labels: list[str],
    correct: torch.Tensor,
    rank_loss: torch.Tensor,
    gap: torch.Tensor,
) -> None:
    correct_values = correct.detach().float().cpu().tolist()
    loss_values = rank_loss.detach().float().cpu().tolist()
    gap_values = gap.detach().float().cpu().tolist()
    for label, is_correct, loss_value, gap_value in zip(labels, correct_values, loss_values, gap_values):
        key = label or "unknown"
        state = groups.setdefault(key, {"count": 0.0, "rank_acc_sum": 0.0, "rank_loss_sum": 0.0, "sim_gap_sum": 0.0})
        state["count"] += 1.0
        state["rank_acc_sum"] += float(is_correct)
        state["rank_loss_sum"] += float(loss_value)
        state["sim_gap_sum"] += float(gap_value)


def finalize_group_stats(groups: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out = {}
    for label, state in sorted(groups.items()):
        count = max(1.0, state["count"])
        out[label] = {
            "count": int(state["count"]),
            "rank_acc": state["rank_acc_sum"] / count,
            "rank_loss": state["rank_loss_sum"] / count,
            "sim_gap": state["sim_gap_sum"] / count,
        }
    return out


def build_model(args: TrainArgs, tokenizer: PreTrainedTokenizerFast) -> RobertaModel:
    config = RobertaConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_len + 2,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        type_vocab_size=1,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    return RobertaModel(config)


def encode(model: nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    output = model(**inputs)
    hidden = output.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1).float() / torch.clamp(mask.sum(dim=1), min=1.0).float()


def tokenize(
    tokenizer: PreTrainedTokenizerFast, texts: list[str], max_len: int, device: torch.device
) -> dict[str, torch.Tensor]:
    batch = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return {key: value.to(device, non_blocking=True) for key, value in batch.items() if key in {"input_ids", "attention_mask"}}


def metrics(
    loss: torch.Tensor,
    pos_loss: torch.Tensor,
    rank_loss: torch.Tensor,
    inbatch_loss: torch.Tensor,
    sigreg_loss: torch.Tensor,
    sim_pos: torch.Tensor,
    sim_neg: torch.Tensor,
    grad_norm: torch.Tensor,
) -> dict[str, float]:
    return {
        "loss": float(loss.detach().cpu()),
        "pos_loss": float(pos_loss.detach().cpu()),
        "rank_loss": float(rank_loss.detach().cpu()),
        "inbatch_loss": float(inbatch_loss.detach().cpu()),
        "sigreg_loss": float(sigreg_loss.detach().cpu()),
        "sim_pos": float(sim_pos.mean().detach().cpu()),
        "sim_neg": float(sim_neg.mean().detach().cpu()),
        "sim_gap": float((sim_pos - sim_neg).mean().detach().cpu()),
        "rank_acc": float((sim_pos > sim_neg).float().mean().detach().cpu()),
        "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
    }


def save_checkpoint(
    out: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
    sigreg: SlicedGaussianRegularizer,
    sampler: ShardSampler,
    eval_sampler: ShardSampler,
    args: TrainArgs,
    step: int,
) -> None:
    payload = {
        "format_version": 2,
        "step": step,
        "model_config": model.config.to_dict(),
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "sigreg": sigreg.state_dict(),
        "sampler": sampler.state_dict(),
        "eval_sampler": eval_sampler.state_dict(),
        "rng": rng_state(),
        "args": asdict(args),
    }
    tmp = out / "latest.pt.tmp"
    latest = out / "latest.pt"
    torch.save(payload, tmp)
    tmp.replace(latest)
    if step and step % (args.save_every * 5) == 0:
        shutil.copy2(latest, out / f"checkpoint-step-{step:08d}.pt")


def resolve_checkpoint_path(value: str, out: Path) -> Path:
    if value == "latest":
        return out / "latest.pt"
    return Path(value).expanduser().resolve()


def load_full_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
    sigreg: SlicedGaussianRegularizer,
    sampler: ShardSampler,
    eval_sampler: ShardSampler,
    device: torch.device,
) -> int:
    checkpoint = torch_load(path, map_location="cpu")
    missing = [key for key in ["optimizer", "scheduler", "sampler", "rng"] if key not in checkpoint]
    if missing:
        raise ValueError(f"{path} is not a full resume checkpoint; missing {missing}. Use --init-checkpoint for model-only init.")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    if "sigreg" in checkpoint:
        sigreg.load_state_dict(checkpoint["sigreg"])
    sampler.load_state_dict(checkpoint["sampler"])
    if "eval_sampler" in checkpoint:
        eval_sampler.load_state_dict(checkpoint["eval_sampler"])
    load_rng_state(checkpoint["rng"], device)
    return int(checkpoint.get("step", 0))


def load_model_checkpoint(path: Path, *, model: nn.Module, device: torch.device) -> int:
    checkpoint = torch_load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    return int(checkpoint.get("step", 0))


def torch_load(path: Path, *, map_location: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def load_rng_state(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def discover_shards(data_roots: list[Path], *, max_shards: int | None) -> list[ShardPair]:
    pairs: list[ShardPair] = []
    for root in data_roots:
        triples_dir = root / "triples"
        views_dir = root / "views"
        if not triples_dir.exists() or not views_dir.exists():
            raise FileNotFoundError(f"missing views/triples under {root}")
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
            ["s5cmd", "sync", "--size-only", f"{out}/", f"{prefix.rstrip('/')}/"],
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
