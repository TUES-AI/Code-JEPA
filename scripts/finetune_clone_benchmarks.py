#!/usr/bin/env python3
"""Fine-tune small Code-JEPA/UniXcoder checkpoints on POJ-104 or BigCloneBench."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code_jepa.models import ENCODER_ONLY, SmallUniXcoder, ensure_unixcoder_special_tokens
from code_jepa.models import unixcoder_tokenize
from scripts.train_codebert_jepa_torch import load_tokenizer, model_from_checkpoint


def _setup_distributed() -> tuple[int, int, int]:
    if "LOCAL_RANK" not in os.environ:
        return 0, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()



def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


@dataclass(frozen=True)
class FineTuneArgs:
    benchmark: str
    benchmark_dir: str
    checkpoint: str
    output_dir: str
    model_name: str = ""
    max_len: int = 256
    batch_size: int = 16
    eval_batch_size: int = 32
    epochs: int = 2
    lr: float = 2e-5
    head_lr: float = 1e-4
    weight_decay: float = 0.01
    margin: float = 0.2
    dropout: float = 0.1
    precision: str = "bf16"
    seed: int = 123456
    max_train_examples: int = 0
    max_valid_examples: int = 0
    max_test_examples: int = 0
    grad_clip: float = 1.0
    freeze_encoder: bool = False
    device: str = "auto"
    log_every: int = 50


class PairClassifier(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        features = torch.cat([left, right, torch.abs(left - right), left * right], dim=-1)
        return self.net(features).squeeze(-1)


def parse_args() -> FineTuneArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["poj104", "bigclonebench"], required=True)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-valid-examples", type=int, default=0)
    parser.add_argument("--max-test-examples", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--log-every", type=int, default=50)
    return FineTuneArgs(**vars(parser.parse_args()))


def main() -> None:
    local_rank, rank, world_size = _setup_distributed()
    args = parse_args()
    set_seed(args.seed + rank)

    out = Path(args.output_dir)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(asdict(args), indent=2, sort_keys=True) + "\n")
    if _is_dist():
        dist.barrier()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_name = args.model_name or str(checkpoint.get("model_name") or "")
    if not model_name:
        raise ValueError("--model-name is required when checkpoint has no model_name")
    tokenizer = load_tokenizer(model_name)
    if checkpoint.get("model_class") == "SmallUniXcoder":
        ensure_unixcoder_special_tokens(tokenizer)
    model = model_from_checkpoint(checkpoint, model_name).to(device)
    model.load_state_dict(checkpoint["ctx_model"])

    if _is_dist():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if rank == 0:
        log(
            out,
            {
                "event": "startup",
                "benchmark": args.benchmark,
                "device": str(device),
                "world_size": world_size,
                "checkpoint_step": int(checkpoint.get("step", -1)),
                "freeze_encoder": args.freeze_encoder,
            },
        )

    if args.benchmark == "bigclonebench":
        report = run_bigclonebench(args, model, tokenizer, device, out, rank=rank, world_size=world_size)
    else:
        report = run_poj104(args, model, tokenizer, device, out, rank=rank, world_size=world_size)

    if rank == 0:
        (out / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))

    if _is_dist():
        dist.destroy_process_group()


def run_bigclonebench(
    args: FineTuneArgs,
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    out: Path,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, Any]:
    base = Path(args.benchmark_dir)
    funcs = load_bigclonebench_functions(resolve_benchmark_file(base, "data.jsonl"))
    all_train_pairs = load_bigclonebench_pairs(
        resolve_benchmark_file(base, "train.txt"),
        funcs,
        max_examples=args.max_train_examples,
        seed=args.seed,
    )
    valid_pairs = load_bigclonebench_pairs(
        resolve_benchmark_file(base, "valid.txt"),
        funcs,
        max_examples=args.max_valid_examples,
        seed=args.seed,
    )
    test_pairs = load_bigclonebench_pairs(
        resolve_benchmark_file(base, "test.txt"),
        funcs,
        max_examples=args.max_test_examples,
        seed=args.seed,
    )

    hidden_size = int(_unwrap(model).config.hidden_size)
    head = PairClassifier(hidden_size, args.dropout).to(device)
    if _is_dist():
        head = DDP(head, device_ids=[device.index], find_unused_parameters=False)
    optimizer = make_optimizer(args, _unwrap(model), _unwrap(head))
    scaler = make_grad_scaler(args, device)
    started = time.time()
    best_valid_f1 = -1.0
    best_threshold = 0.5

    for epoch in range(1, args.epochs + 1):
        rng = random.Random(args.seed + epoch)
        rng.shuffle(all_train_pairs)
        # Each rank processes its own shard of the training data
        train_pairs = all_train_pairs[rank::world_size]
        train_metrics = train_bigclonebench_epoch(
            model,
            head,
            tokenizer,
            optimizer,
            scaler,
            train_pairs,
            args,
            device,
            out,
            epoch,
            rank=rank,
        )
        if rank == 0:
            valid_metrics = evaluate_bigclonebench(_unwrap(model), _unwrap(head), tokenizer, valid_pairs, args, device)
            if valid_metrics["best_f1"] > best_valid_f1:
                best_valid_f1 = valid_metrics["best_f1"]
                best_threshold = valid_metrics["best_threshold"]
            log(
                out,
                {
                    "event": "epoch",
                    "epoch": epoch,
                    "elapsed_s": round(time.time() - started, 2),
                    **prefix_keys("train_", train_metrics),
                    **prefix_keys("valid_", valid_metrics),
                },
            )
        if _is_dist():
            dist.barrier()

    test_metrics: dict[str, Any] = {}
    if rank == 0:
        test_metrics = evaluate_bigclonebench(
            _unwrap(model), _unwrap(head), tokenizer, test_pairs, args, device, threshold=best_threshold,
        )
        save_finetuned(out, _unwrap(model), args, task_state={"pair_head": _unwrap(head).state_dict()})
    return {
        "benchmark": "bigclonebench",
        "train_pairs": len(all_train_pairs),
        "valid_pairs": len(valid_pairs),
        "test_pairs": len(test_pairs),
        "best_valid_f1": best_valid_f1,
        "valid_selected_threshold": best_threshold,
        "test": test_metrics,
    }


def train_bigclonebench_epoch(
    model: nn.Module,
    head: nn.Module,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    pairs: list[tuple[str, str, int]],
    args: FineTuneArgs,
    device: torch.device,
    out: Path,
    epoch: int,
    *,
    rank: int = 0,
) -> dict[str, float]:
    model.train(not args.freeze_encoder)
    head.train()
    losses: list[float] = []
    amp_dtype = autocast_dtype(args.precision)
    for step, batch_pairs in enumerate(chunks(pairs, args.batch_size), start=1):
        left = [item[0] for item in batch_pairs]
        right = [item[1] for item in batch_pairs]
        labels = torch.tensor([item[2] for item in batch_pairs], dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=(device.type == "cuda" and amp_dtype is not None),
        ):
            left_h, right_h = encode_pair_for_training(
                model, tokenizer, left, right, args, device
            )
            logits = head(left_h, right_h)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_parameters(model, head, args), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
        if rank == 0 and (step == 1 or step % args.log_every == 0):
            log(out, {"event": "train_batch", "epoch": epoch, "step": step, "loss": losses[-1]})
    return {"loss": float(np.mean(losses)) if losses else 0.0}


@torch.no_grad()
def evaluate_bigclonebench(
    model: nn.Module,
    head: PairClassifier,
    tokenizer: Any,
    pairs: list[tuple[str, str, int]],
    args: FineTuneArgs,
    device: torch.device,
    *,
    threshold: float | None = None,
) -> dict[str, float]:
    model.eval()
    head.eval()
    scores: list[float] = []
    labels: list[int] = []
    for batch_pairs in chunks(pairs, args.eval_batch_size):
        left = [item[0] for item in batch_pairs]
        right = [item[1] for item in batch_pairs]
        left_h, right_h = encode_pair_for_training(model, tokenizer, left, right, args, device)
        probs = torch.sigmoid(head(left_h, right_h)).float().cpu().numpy()
        scores.extend(float(x) for x in probs)
        labels.extend(int(item[2]) for item in batch_pairs)
    return binary_metrics(np.asarray(labels, dtype=np.int64), np.asarray(scores), threshold)


def _load_poj_splits(args: FineTuneArgs) -> tuple[list[dict], list[dict], list[dict]]:
    """Load POJ-104 splits from local jsonl files or HuggingFace if files are missing."""
    base = Path(args.benchmark_dir)
    local_train = base / "train.jsonl"
    local_valid = base / "valid.jsonl"
    local_test = base / "test.jsonl"

    if local_train.exists() and local_valid.exists() and local_test.exists():
        return (
            load_poj_rows(local_train, 0),
            load_poj_rows(local_valid, args.max_valid_examples),
            load_poj_rows(local_test, args.max_test_examples),
        )

    from datasets import load_dataset  # type: ignore
    ds = load_dataset("google/code_x_glue_cc_clone_detection_poj104")

    def hf_to_rows(split: str, max_examples: int) -> list[dict]:
        rows = [{"code": str(r["code"]), "label": str(r["label"])} for r in ds[split]]
        if max_examples > 0:
            rows = rows[:max_examples]
        return rows

    return (
        hf_to_rows("train", 0),
        hf_to_rows("validation", args.max_valid_examples),
        hf_to_rows("test", args.max_test_examples),
    )


def run_poj104(
    args: FineTuneArgs,
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    out: Path,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, Any]:
    all_train_rows, valid_rows, test_rows = _load_poj_splits(args)
    # Each rank trains on its own shard (by label-preserving stride)
    train_rows = all_train_rows[rank::world_size]
    optimizer = make_optimizer(args, _unwrap(model), None)
    scaler = make_grad_scaler(args, device)
    started = time.time()
    best_valid_mapr = -1.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_poj_epoch(
            model,
            tokenizer,
            optimizer,
            scaler,
            train_rows,
            args,
            device,
            out,
            epoch,
            rank=rank,
        )
        if rank == 0:
            valid_metrics = evaluate_poj_mapr(_unwrap(model), tokenizer, valid_rows, args, device)
            best_valid_mapr = max(best_valid_mapr, valid_metrics["map_at_r"])
            log(
                out,
                {
                    "event": "epoch",
                    "epoch": epoch,
                    "elapsed_s": round(time.time() - started, 2),
                    **prefix_keys("train_", train_metrics),
                    **prefix_keys("valid_", valid_metrics),
                },
            )
        if _is_dist():
            dist.barrier()

    test_metrics: dict[str, Any] = {}
    if rank == 0:
        test_metrics = evaluate_poj_mapr(_unwrap(model), tokenizer, test_rows, args, device)
        save_finetuned(out, _unwrap(model), args, task_state={})
    return {
        "benchmark": "poj104",
        "train_rows": len(all_train_rows),
        "valid_rows": len(valid_rows),
        "test_rows": len(test_rows),
        "best_valid_map_at_r": best_valid_mapr,
        "test": test_metrics,
    }


def train_poj_epoch(
    model: nn.Module,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    rows: list[dict[str, Any]],
    args: FineTuneArgs,
    device: torch.device,
    out: Path,
    epoch: int,
    *,
    rank: int = 0,
) -> dict[str, float]:
    model.train(not args.freeze_encoder)
    sampler = PojTripletSampler(rows, seed=args.seed + epoch)
    triplets = args.max_train_examples or len(rows)
    steps = max(1, math.ceil(triplets / args.batch_size))
    amp_dtype = autocast_dtype(args.precision)
    losses: list[float] = []
    accuracies: list[float] = []
    for step in range(1, steps + 1):
        anchors, positives, negatives = sampler.next_batch(args.batch_size)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=(device.type == "cuda" and amp_dtype is not None),
        ):
            encoded = encode_for_training(
                model,
                tokenizer,
                anchors + positives + negatives,
                args,
                device,
            )
            za, zp, zn = encoded.chunk(3, dim=0)
            za = F.normalize(za, dim=-1)
            zp = F.normalize(zp, dim=-1)
            zn = F.normalize(zn, dim=-1)
            sim_pos = torch.sum(za * zp, dim=-1)
            sim_neg = torch.sum(za * zn, dim=-1)
            loss = F.relu(args.margin + sim_neg - sim_pos).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_parameters(model, None, args), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
        accuracies.append(float((sim_pos > sim_neg).float().mean().detach().cpu()))
        if rank == 0 and (step == 1 or step % args.log_every == 0):
            log(
                out,
                {
                    "event": "train_batch",
                    "epoch": epoch,
                    "step": step,
                    "loss": losses[-1],
                    "triplet_acc": accuracies[-1],
                },
            )
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "triplet_acc": float(np.mean(accuracies)) if accuracies else 0.0,
    }


@torch.no_grad()
def evaluate_poj_mapr(
    model: nn.Module,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    args: FineTuneArgs,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    labels = np.asarray([str(row["label"]) for row in rows])
    embeddings = encode_inference(
        model,
        tokenizer,
        [str(row["code"]) for row in rows],
        args.max_len,
        args.eval_batch_size,
        device,
    ).to(device)
    embeddings = F.normalize(embeddings, dim=-1)
    counts = Counter(labels)
    aps: list[float] = []
    query_batch = max(1, min(args.eval_batch_size, 256))
    for start in range(0, len(rows), query_batch):
        end = min(start + query_batch, len(rows))
        scores = embeddings[start:end] @ embeddings.T
        diag = torch.arange(start, end, device=device)
        scores[torch.arange(end - start, device=device), diag] = -torch.inf
        order = torch.argsort(scores, dim=1, descending=True).cpu().numpy()
        for offset, ranked in enumerate(order):
            label = labels[start + offset]
            r = counts[label] - 1
            if r <= 0:
                continue
            top_r = ranked[:r]
            relevant = (labels[top_r] == label).astype(np.float32)
            precision = np.cumsum(relevant) / (np.arange(r) + 1)
            aps.append(float((precision * relevant).sum() / r))
    return {
        "map_at_r": float(np.mean(aps)) if aps else 0.0,
        "queries": float(len(aps)),
        "rows": float(len(rows)),
    }


class PojTripletSampler:
    def __init__(self, rows: list[dict[str, Any]], *, seed: int) -> None:
        self.rows = rows
        self.rng = random.Random(seed)
        self.by_label: dict[str, list[int]] = defaultdict(list)
        for idx, row in enumerate(rows):
            self.by_label[str(row["label"])].append(idx)
        self.labels = [label for label, values in self.by_label.items() if len(values) > 1]
        if len(self.labels) < 2:
            raise RuntimeError("POJ-104 training requires at least two labels with >=2 examples")

    def next_batch(self, batch_size: int) -> tuple[list[str], list[str], list[str]]:
        anchors: list[str] = []
        positives: list[str] = []
        negatives: list[str] = []
        for _ in range(batch_size):
            label = self.rng.choice(self.labels)
            anchor_idx = self.rng.choice(self.by_label[label])
            positive_idx = self.rng.choice(self.by_label[label])
            while positive_idx == anchor_idx:
                positive_idx = self.rng.choice(self.by_label[label])
            negative_label = self.rng.choice(self.labels)
            while negative_label == label:
                negative_label = self.rng.choice(self.labels)
            negative_idx = self.rng.choice(self.by_label[negative_label])
            anchors.append(str(self.rows[anchor_idx]["code"]))
            positives.append(str(self.rows[positive_idx]["code"]))
            negatives.append(str(self.rows[negative_idx]["code"]))
        return anchors, positives, negatives


def encode_pair_for_training(
    model: nn.Module,
    tokenizer: Any,
    left: list[str],
    right: list[str],
    args: FineTuneArgs,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = encode_for_training(model, tokenizer, left + right, args, device)
    return encoded.chunk(2, dim=0)


def encode_for_training(
    model: nn.Module,
    tokenizer: Any,
    texts: list[str],
    args: FineTuneArgs,
    device: torch.device,
) -> torch.Tensor:
    if args.freeze_encoder:
        with torch.no_grad():
            return encode_batch(model, tokenizer, texts, args.max_len, device).detach()
    return encode_batch(model, tokenizer, texts, args.max_len, device)


@torch.no_grad()
def encode_inference(
    model: nn.Module,
    tokenizer: Any,
    texts: list[str],
    max_len: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for batch in chunks(texts, batch_size):
        outputs.append(encode_batch(model, tokenizer, list(batch), max_len, device).float().cpu())
    return torch.cat(outputs, dim=0)


def encode_batch(
    model: nn.Module,
    tokenizer: Any,
    texts: list[str],
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    max_len = effective_max_len(model, max_len)
    if isinstance(model, SmallUniXcoder):
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
    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    hidden = model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    return (torch.sum(hidden * mask, dim=1) / torch.clamp(torch.sum(mask, dim=1), min=1.0)).float()


def load_bigclonebench_functions(path: Path) -> dict[str, str]:
    funcs: dict[str, str] = {}
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            funcs[str(row["idx"])] = str(row["func"])
    return funcs


def load_bigclonebench_pairs(
    path: Path,
    funcs: dict[str, str],
    *,
    max_examples: int,
    seed: int,
) -> list[tuple[str, str, int]]:
    pairs: list[tuple[str, str, int]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            left, right, label = parts
            if left not in funcs or right not in funcs:
                continue
            pairs.append((funcs[left], funcs[right], int(label)))
    if max_examples > 0 and len(pairs) > max_examples:
        rng = random.Random(seed)
        rng.shuffle(pairs)
        pairs = pairs[:max_examples]
    return pairs


def load_poj_rows(path: Path, max_examples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("code") is None or row.get("label") is None:
                continue
            rows.append(row)
            if max_examples > 0 and len(rows) >= max_examples:
                break
    return rows


def resolve_benchmark_file(base: Path, name: str) -> Path:
    candidates = [base / name, base / "dataset" / name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing {name}; checked {', '.join(str(x) for x in candidates)}")


def binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float]:
    if len(labels) == 0:
        return {}
    best = best_f1(labels, scores)
    selected = best["threshold"] if threshold is None else threshold
    selected_metrics = f1_at_threshold(labels, scores, selected)
    return {
        "average_precision": average_precision(labels, scores),
        "roc_auc": roc_auc(labels, scores),
        "best_f1": best["f1"],
        "best_threshold": best["threshold"],
        "f1_at_selected_threshold": selected_metrics["f1"],
        "precision_at_selected_threshold": selected_metrics["precision"],
        "recall_at_selected_threshold": selected_metrics["recall"],
        "accuracy_at_selected_threshold": selected_metrics["accuracy"],
        "selected_threshold": selected,
        "positive_rate": float(labels.mean()),
        "mean_positive_score": float(scores[labels == 1].mean()),
        "mean_negative_score": float(scores[labels == 0].mean()),
    }


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    positives = sorted_labels.sum()
    if positives == 0:
        return 0.0
    precision = np.cumsum(sorted_labels) / (np.arange(len(sorted_labels)) + 1)
    return float((precision * sorted_labels).sum() / positives)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.0
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    positive_rank_sum = ranks[labels == 1].sum()
    auc = (positive_rank_sum - len(positives) * (len(positives) + 1) / 2) / (
        len(positives) * len(negatives)
    )
    return float(auc)


def best_f1(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    best = {"f1": 0.0, "threshold": 0.5}
    for threshold in np.unique(scores):
        metrics = f1_at_threshold(labels, scores, float(threshold))
        if metrics["f1"] > best["f1"]:
            best = {"f1": metrics["f1"], "threshold": float(threshold)}
    return best


def f1_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (scores >= threshold).astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    accuracy = (tp + tn) / max(1, len(labels))
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def make_optimizer(args: FineTuneArgs, model: nn.Module, head: nn.Module | None) -> AdamW:
    groups: list[dict[str, Any]] = []
    if not args.freeze_encoder:
        groups.append({"params": list(model.parameters()), "lr": args.lr})
    if head is not None:
        groups.append({"params": list(head.parameters()), "lr": args.head_lr})
    if not groups:
        raise ValueError("nothing to optimize; do not freeze encoder without a task head")
    if torch.cuda.is_available():
        try:
            return AdamW(groups, weight_decay=args.weight_decay, fused=True)
        except TypeError:
            pass
    return AdamW(groups, weight_decay=args.weight_decay)


def make_grad_scaler(args: FineTuneArgs, device: torch.device) -> Any:
    enabled = device.type == "cuda" and args.precision == "fp16"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def trainable_parameters(
    model: nn.Module,
    head: nn.Module | None,
    args: FineTuneArgs,
) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    if not args.freeze_encoder:
        params.extend(model.parameters())
    if head is not None:
        params.extend(head.parameters())
    return params


def save_finetuned(
    out: Path,
    model: nn.Module,
    args: FineTuneArgs,
    *,
    task_state: dict[str, Any],
) -> None:
    payload = {
        "args": asdict(args),
        "model_class": "SmallUniXcoder" if isinstance(model, SmallUniXcoder) else type(model).__name__,
        "model_config": model.config.to_dict() if hasattr(model, "config") else {},
        "ctx_model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "task_state": to_cpu_state(task_state),
    }
    torch.save(payload, out / "finetuned.pt")


def to_cpu_state(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_cpu_state(inner) for key, inner in value.items()}
    if torch.is_tensor(value):
        return value.detach().cpu()
    return value


def effective_max_len(model: nn.Module, requested: int) -> int:
    config_max = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if config_max is None:
        return requested
    return max(4, min(requested, int(config_max) - 2))


def choose_device(name: str) -> torch.device:
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def autocast_dtype(precision: str) -> torch.dtype | None:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def chunks(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    chunk: list[Any] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def prefix_keys(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in values.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log(out: Path, record: dict[str, Any]) -> None:
    record = dict(record)
    record.setdefault("time", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


if __name__ == "__main__":
    main()
