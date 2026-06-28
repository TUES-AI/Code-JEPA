#!/usr/bin/env python3
"""Train a CodeBERT/RoBERTa JEPA ranker over Code-JEPA Parquet triples.

This is the first real backbone trainer: one trainable RoBERTa/CodeBERT
encoder sees all views, SIGReg regularizes encoder outputs, and positives/negatives
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
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformers import AutoConfig, AutoModel, AutoTokenizer, RobertaTokenizerFast
from transformers import get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from code_jepa.models import (
    ENCODER_ONLY,
    SlicedGaussianRegularizer,
    SmallUniXcoder,
    count_parameters,
    ensure_unixcoder_special_tokens,
    small_unixcoder_config,
    unixcoder_tokenize,
)

import os as _os


def _setup_distributed() -> tuple[int, int, int]:
    """Init NCCL process group when running under torchrun. Returns (local_rank, rank, world_size)."""
    if "LOCAL_RANK" not in _os.environ:
        return 0, 0, 1
    local_rank = int(_os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0



def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


@dataclass(frozen=True)
class TrainArgs:
    data_roots: list[str]
    output_dir: str
    model_name: str = "microsoft/codebert-base"
    init: str = "pretrained"
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
    sigreg_weight: float = 0.05
    sigreg_slices: int = 64
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
    def __init__(
        self,
        pairs: list[ShardPair],
        view_by_id: dict[str, str],
        *,
        seed: int,
    ) -> None:
        self.pairs = pairs
        self.view_by_id = view_by_id
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
        triple_table = pq.read_table(
            pair.triples_path,
            columns=["anchor_view_id", "positive_view_id", "negative_view_id"],
        )
        anchor_ids = triple_table.column("anchor_view_id").to_pylist()
        positive_ids = triple_table.column("positive_view_id").to_pylist()
        negative_ids = triple_table.column("negative_view_id").to_pylist()

        examples: list[tuple[str, str, str]] = []
        for anchor_id, positive_id, negative_id in zip(anchor_ids, positive_ids, negative_ids):
            anchor = self.view_by_id.get(anchor_id)
            positive = self.view_by_id.get(positive_id)
            negative = self.view_by_id.get(negative_id)
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
    parser.add_argument(
        "--init",
        choices=["pretrained", "scratch", "roberta_large_scratch", "unixcoder_small_scratch"],
        default="pretrained",
    )
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
    parser.add_argument("--sigreg-weight", type=float, default=0.05)
    parser.add_argument("--sigreg-slices", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
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
    local_rank, rank, world_size = _setup_distributed()
    args = parse_args()
    set_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    out = Path(args.output_dir)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(
            json.dumps(asdict(args), indent=2, sort_keys=True) + "\n"
        )
    if _is_dist():
        dist.barrier()

    data_roots = [Path(root) for root in args.data_roots]
    pairs = discover_shards(data_roots, max_shards=args.max_shards)
    if rank == 0:
        log(out, {"event": "loading_views", "data_roots": args.data_roots})
    view_by_id = load_all_views(data_roots)
    if rank == 0:
        log(
            out,
            {
                "event": "startup",
                "pairs": len(pairs),
                "views_loaded": len(view_by_id),
                "world_size": world_size,
                "args": asdict(args),
                "device": device_name(),
            },
        )

    if args.eval_only:
        if rank == 0:
            metrics = evaluate_checkpoint(args, pairs, view_by_id)
            log(out, {"event": "eval_only", **metrics})
        return

    train(args, pairs, view_by_id, out, local_rank=local_rank)


def train(args: TrainArgs, pairs: list[ShardPair], view_by_id: dict[str, str], out: Path, *, local_rank: int = 0) -> None:
    rank = _rank()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(args.model_name)
    if args.init == "unixcoder_small_scratch":
        added = ensure_unixcoder_special_tokens(tokenizer)
        if added and rank == 0:
            log(out, {"event": "tokenizer_added_unixcoder_tokens", "count": added})
    ctx = build_model(args, tokenizer).to(device)
    hidden_size = int(ctx.config.hidden_size)
    predictor = Predictor(hidden_size, args.dropout).to(device)
    sigreg = SlicedGaussianRegularizer(num_slices=args.sigreg_slices).to(device)
    parameter_count = count_parameters(ctx)
    if args.init == "unixcoder_small_scratch":
        validate_small_unixcoder_size(parameter_count, tokenizer)
    if rank == 0:
        log(
            out,
            {
                "event": "model_built",
                "model_class": model_class_name(ctx),
                "unique_parameters": parameter_count,
                "hidden_size": hidden_size,
            },
        )

    # Gradient checkpointing is incompatible with static_graph DDP; skip it in distributed runs
    if args.gradient_checkpointing and not _is_dist() and hasattr(ctx, "gradient_checkpointing_enable"):
        ctx.gradient_checkpointing_enable()

    if _is_dist():
        ctx = DDP(ctx, device_ids=[local_rank], find_unused_parameters=True)
        predictor = DDP(predictor, device_ids=[local_rank], find_unused_parameters=True)

    if args.compile:
        predictor = torch.compile(predictor)  # type: ignore[assignment]

    optimizer = make_optimizer(list(_unwrap(ctx).parameters()) + list(_unwrap(predictor).parameters()), args)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.precision == "fp16"))
    # Each rank uses a different seed so they sample different shards
    sampler = ShardSampler(pairs, view_by_id, seed=args.seed + rank)
    eval_sampler = ShardSampler(pairs, view_by_id, seed=args.seed + 17 + rank)
    started = time.time()
    deadline = started + args.duration_hours * 3600 if args.duration_hours > 0 else math.inf

    for step in range(1, args.steps + 1):
        if time.time() >= deadline:
            if rank == 0:
                log(out, {"event": "deadline", "step": step})
            break
        batch_started = time.time()
        batch = sampler.next_batch(args.batch_size)
        metrics = train_step(
            ctx,
            predictor,
            sigreg,
            tokenizer,
            optimizer,
            scheduler,
            scaler,
            batch,
            args,
            device,
        )
        elapsed = time.time() - started
        step_s = time.time() - batch_started
        remaining_steps = args.steps - step
        eta_h = round(remaining_steps * (elapsed / step) / 3600, 2) if step > 0 else 0
        if rank == 0:
            metrics.update(
                {
                    "event": "train",
                    "step": step,
                    "elapsed_s": round(elapsed, 2),
                    "batch_s": round(step_s, 3),
                    "eta_h": eta_h,
                    "lr": scheduler.get_last_lr()[0],
                    "shard": sampler.current.name if sampler.current else "",
                    "shard_load_s": round(sampler.last_load_s, 3),
                }
            )
            if step == 1 or step % args.log_every == 0:
                log(out, metrics)
        if args.dry_run_batches and step >= args.dry_run_batches:
            if rank == 0:
                log(out, {"event": "dry_run_done", "step": step})
            break
        if args.eval_every > 0 and step % args.eval_every == 0 and rank == 0:
            eval_metrics = evaluate_batches(_unwrap(ctx), _unwrap(predictor), tokenizer, eval_sampler, args, device)
            log(out, {"event": "eval", "step": step, **eval_metrics})
        if args.save_every > 0 and step % args.save_every == 0 and rank == 0:
            save_light_checkpoint(out, _unwrap(ctx), _unwrap(predictor), args, step)
        if args.s3_output_prefix and args.s3_sync_every > 0 and step % args.s3_sync_every == 0 and rank == 0:
            sync_s3(out, args.s3_output_prefix)

    if rank == 0:
        save_light_checkpoint(out, _unwrap(ctx), _unwrap(predictor), args, step)
        if args.s3_output_prefix:
            sync_s3(out, args.s3_output_prefix)
        log(out, {"event": "done", "step": step, "elapsed_s": round(time.time() - started, 2)})
    if _is_dist():
        dist.destroy_process_group()


def train_step(
    ctx: nn.Module,
    predictor: nn.Module,
    sigreg: SlicedGaussianRegularizer,
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
    unixcoder_mode = isinstance(ctx, SmallUniXcoder)
    anchor_inputs = tokenize(
        tokenizer,
        anchors,
        args.max_len,
        device,
        unixcoder_mode=unixcoder_mode,
    )
    view_inputs = tokenize(
        tokenizer,
        positives + negatives,
        args.max_len,
        device,
        unixcoder_mode=unixcoder_mode,
    )
    amp_dtype = autocast_dtype(args.precision)

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=(device.type == "cuda" and amp_dtype is not None),
    ):
        za = encode(ctx, anchor_inputs)
        zp, zn = encode(ctx, view_inputs).chunk(2, dim=0)
        pred = predictor(za)
        pred_n = F.normalize(pred, dim=-1)
        zp_n = F.normalize(zp, dim=-1)
        zn_n = F.normalize(zn, dim=-1)
        sim_pos = torch.sum(pred_n * zp_n, dim=-1)
        sim_neg = torch.sum(pred_n * zn_n, dim=-1)
        jepa_loss = F.mse_loss(pred, zp)
        rank_loss = F.relu(args.margin + sim_neg - sim_pos).mean()
        logits = pred_n @ zp_n.T / args.temperature
        labels = torch.arange(pred.shape[0], device=pred.device)
        inbatch_loss = F.cross_entropy(logits, labels)
        sigreg_loss = sigreg(torch.cat([za, zp, zn], dim=0))
        loss = (
            jepa_loss
            + args.rank_weight * rank_loss
            + args.inbatch_weight * inbatch_loss
            + args.sigreg_weight * sigreg_loss
        )

    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite loss: {float(loss.detach().cpu())}")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = float(
        torch.nn.utils.clip_grad_norm_(
            list(ctx.parameters()) + list(predictor.parameters()), args.grad_clip
        )
        .detach()
        .cpu()
    )
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    rank_acc = (sim_pos > sim_neg).float().mean()
    return {
        "loss": float(loss.detach().cpu()),
        "jepa_loss": float(jepa_loss.detach().cpu()),
        "rank_loss": float(rank_loss.detach().cpu()),
        "inbatch_loss": float(inbatch_loss.detach().cpu()),
        "sigreg_loss": float(sigreg_loss.detach().cpu()),
        "sim_pos": float(sim_pos.mean().detach().cpu()),
        "sim_neg": float(sim_neg.mean().detach().cpu()),
        "rank_acc": float(rank_acc.detach().cpu()),
        "grad_norm": round(grad_norm, 4),
        "grad_clipped": grad_norm > args.grad_clip,
        "grad_exploding": grad_norm > 5.0 * args.grad_clip,
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
    unixcoder_mode = isinstance(model, SmallUniXcoder)
    for _ in range(args.eval_batches):
        anchors, positives, negatives = sampler.next_batch(args.batch_size)
        anchor_inputs = tokenize(
            tokenizer,
            anchors,
            args.max_len,
            device,
            unixcoder_mode=unixcoder_mode,
        )
        view_inputs = tokenize(
            tokenizer,
            positives + negatives,
            args.max_len,
            device,
            unixcoder_mode=unixcoder_mode,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=(device.type == "cuda" and amp_dtype is not None),
        ):
            pred = F.normalize(predictor(encode(model, anchor_inputs)), dim=-1)
            zp, zn = encode(model, view_inputs).chunk(2, dim=0)
            zp = F.normalize(zp, dim=-1)
            zn = F.normalize(zn, dim=-1)
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


def evaluate_checkpoint(args: TrainArgs, pairs: list[ShardPair], view_by_id: dict[str, str]) -> dict[str, float]:
    if not args.checkpoint:
        raise ValueError("--eval-only requires --checkpoint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_name = checkpoint.get("model_name", args.model_name)
    tokenizer = load_tokenizer(model_name)
    if (
        checkpoint.get("model_class") == "SmallUniXcoder"
        or args.init == "unixcoder_small_scratch"
    ):
        ensure_unixcoder_special_tokens(tokenizer)
    model = model_from_checkpoint(checkpoint, model_name).to(device)
    hidden_size = int(model.config.hidden_size)
    predictor = Predictor(hidden_size, args.dropout).to(device)
    model.load_state_dict(checkpoint["ctx_model"])
    predictor.load_state_dict(checkpoint["predictor"])
    sampler = ShardSampler(pairs, view_by_id, seed=args.seed + 101)
    return evaluate_batches(model, predictor, tokenizer, sampler, args, device)


def build_model(args: TrainArgs, tokenizer: Any) -> nn.Module:
    if args.init == "pretrained":
        return AutoModel.from_pretrained(args.model_name)
    if args.init == "unixcoder_small_scratch":
        config = small_unixcoder_config(
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id or tokenizer.cls_token_id,
            eos_token_id=tokenizer.eos_token_id or tokenizer.sep_token_id,
            max_position_embeddings=args.max_len + 2,
        )
        return SmallUniXcoder(config)
    config = AutoConfig.from_pretrained(args.model_name)
    if args.init == "roberta_large_scratch":
        if getattr(config, "model_type", "") != "roberta":
            raise ValueError(
                "--init roberta_large_scratch requires a RoBERTa/CodeBERT-style config"
            )
        config.hidden_size = 1024
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
        config.intermediate_size = 4096
        if hasattr(config, "type_vocab_size"):
            config.type_vocab_size = 1
    return AutoModel.from_config(config)


def model_from_checkpoint(checkpoint: dict[str, Any], fallback_model_name: str) -> nn.Module:
    config_dict = checkpoint.get("model_config")
    if config_dict:
        model_type = config_dict.get("model_type")
        config = (
            AutoConfig.for_model(model_type)
            if model_type
            else AutoConfig.from_pretrained(fallback_model_name)
        )
        config.update(config_dict)
        if checkpoint.get("model_class") == "SmallUniXcoder":
            return SmallUniXcoder(config)
        return AutoModel.from_config(config)
    return AutoModel.from_pretrained(fallback_model_name)


def encode(model: nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    output = model(**inputs)
    hidden = output.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = torch.sum(hidden * mask, dim=1) / torch.clamp(torch.sum(mask, dim=1), min=1.0)
    return pooled.float()


def tokenize(
    tokenizer: Any,
    texts: list[str],
    max_len: int,
    device: torch.device,
    *,
    unixcoder_mode: bool = False,
) -> dict[str, torch.Tensor]:
    if unixcoder_mode:
        batch = unixcoder_tokenize(
            tokenizer,
            texts,
            mode=ENCODER_ONLY,
            padding="longest",
            max_length=max_len,
            return_tensors="pt",
        )
    else:
        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def save_light_checkpoint(
    out: Path, ctx: nn.Module, predictor: nn.Module, args: TrainArgs, step: int
) -> None:
    payload = {
        "step": step,
        "model_name": args.model_name,
        "model_class": model_class_name(ctx),
        "model_config": ctx.config.to_dict() if hasattr(ctx, "config") else {},
        "ctx_model": {k: v.detach().cpu() for k, v in ctx.state_dict().items()},
        "predictor": {k: v.detach().cpu() for k, v in predictor.state_dict().items()},
        "args": asdict(args),
    }
    tmp = out / "latest.pt.tmp"
    latest = out / "latest.pt"
    torch.save(payload, tmp)
    tmp.replace(latest)
    if args.save_every > 0 and step % (args.save_every * 5) == 0:
        shutil.copy2(latest, out / f"checkpoint-step-{step:08d}.pt")


def make_optimizer(params: list[nn.Parameter], args: TrainArgs) -> torch.optim.Optimizer:
    if torch.cuda.is_available():
        try:
            return AdamW(params, lr=args.lr, weight_decay=args.weight_decay, fused=True)
        except TypeError:
            pass
    return AdamW(params, lr=args.lr, weight_decay=args.weight_decay)


def model_class_name(model: nn.Module) -> str:
    if isinstance(model, SmallUniXcoder):
        return "SmallUniXcoder"
    return type(model).__name__


def load_tokenizer(model_name_or_path: str) -> Any:
    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    except Exception:
        path = Path(model_name_or_path)
        vocab_file = path / "vocab.json"
        merges_file = path / "merges.txt"
        if not vocab_file.exists() or not merges_file.exists():
            raise
        return RobertaTokenizerFast(
            vocab_file=str(vocab_file),
            merges_file=str(merges_file),
            bos_token="<bos>",
            eos_token="<eos>",
            unk_token="<unk>",
            pad_token="<pad>",
            add_prefix_space=False,
            model_max_length=512,
        )


def validate_small_unixcoder_size(parameter_count: int, tokenizer: Any) -> None:
    if 25_000_000 <= parameter_count <= 30_000_000:
        return
    raise ValueError(
        "--init unixcoder_small_scratch is intended for the 25-30M small tier, "
        f"but this tokenizer gives {parameter_count:,} parameters with vocab size "
        f"{len(tokenizer):,}. Use the Code-JEPA 16k BPE tokenizer via --model-name."
    )


def load_all_views(data_roots: list[Path]) -> dict[str, str]:
    """Load all views from all data roots into a global view_id -> code mapping.

    Views are ~2 GB on disk across all stages and must be loaded globally because
    triples shards reference view IDs from across all views shards — same-numbered
    shard pairing does not hold.
    """
    view_by_id: dict[str, str] = {}
    for root in data_roots:
        views_dir = root / "views"
        if not views_dir.exists():
            raise FileNotFoundError(f"missing views dir: {views_dir}")
        for views_path in sorted(views_dir.rglob("*.parquet")):
            table = pq.read_table(str(views_path), columns=["view_id", "code"])
            for view_id, code in zip(
                table.column("view_id").to_pylist(),
                table.column("code").to_pylist(),
            ):
                if view_id and code:
                    view_by_id[view_id] = code
    return view_by_id


def discover_shards(data_roots: list[Path], *, max_shards: int | None) -> list[ShardPair]:
    pairs: list[ShardPair] = []
    for root in data_roots:
        triples_dir = root / "triples"
        if not triples_dir.exists():
            raise FileNotFoundError(f"missing triples dir: {triples_dir}")
        for triples_path in sorted(triples_dir.rglob("*.parquet")):
            rel = triples_path.relative_to(triples_dir)
            pairs.append(
                ShardPair(
                    root=str(root),
                    name=f"{root.name}/{rel.as_posix()}",
                    triples_path=str(triples_path),
                )
            )
    if not pairs:
        raise FileNotFoundError(f"no triples shards under {data_roots}")
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
    # Always write to metrics.jsonl (primary log, safe for parallel jobs)
    with (out / "metrics.jsonl").open("a") as f:
        f.write(line + "\n")
        f.flush()
    # Print non-train events to stdout so SLURM log shows progress milestones
    event = record.get("event", "")
    if event != "train" or record.get("grad_exploding"):
        print(line, flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
