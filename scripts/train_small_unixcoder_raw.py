#!/usr/bin/env python3
"""Pretrain the small UniXcoder-style RoBERTa backbone on raw CodeSearchNet rows.

This is a baseline pretrainer for matching Code-JEPA's small model scale while
using raw CodeSearchNet function/docstring pairs instead of JEPA views/triples.
It trains the same ``SmallUniXcoder`` backbone with a RoBERTa-style MLM loss and
an optional code-doc contrastive loss.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from code_jepa.models import (  # noqa: E402
    ENCODER_ONLY,
    SmallUniXcoder,
    count_parameters,
    ensure_unixcoder_special_tokens,
    small_unixcoder_config,
    unixcoder_tokenize,
)


@dataclass(frozen=True)
class Args:
    output_dir: str
    model_name: str = "assets/tokenizers/codesearchnet-python/bpe16k"
    dataset_name: str = "code_search_net"
    dataset_config: str = "python"
    split: str = "train"
    local_dataset_dir: str = ""
    data_files: list[str] | None = None
    streaming: bool = True
    max_len: int = 256
    batch_size: int = 128
    steps: int = 200_000
    duration_hours: float = 0.0
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 2_000
    mlm_probability: float = 0.15
    contrastive_weight: float = 0.1
    temperature: float = 0.05
    precision: str = "bf16"
    seed: int = 123456
    log_every: int = 20
    eval_every: int = 1_000
    eval_batches: int = 20
    save_every: int = 5_000
    shuffle_buffer: int = 10_000
    min_code_tokens: int = 8
    min_doc_tokens: int = 3
    max_rows_in_memory: int = 0
    dry_run_batches: int = 0


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="assets/tokenizers/codesearchnet-python/bpe16k")
    parser.add_argument("--dataset-name", default="code_search_net")
    parser.add_argument("--dataset-config", default="python")
    parser.add_argument("--split", default="train")
    parser.add_argument("--local-dataset-dir", default="")
    parser.add_argument("--data-files", nargs="+", default=None)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--duration-hours", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--min-code-tokens", type=int, default=8)
    parser.add_argument("--min-doc-tokens", type=int, default=3)
    parser.add_argument("--max-rows-in-memory", type=int, default=0)
    parser.add_argument("--dry-run-batches", type=int, default=0)
    return Args(**vars(parser.parse_args()))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(args), indent=2, sort_keys=True) + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    ensure_unixcoder_special_tokens(tokenizer)
    model = build_model(args, tokenizer).to(device)
    param_count = count_parameters(model)
    log(out, {"event": "model_built", "unique_parameters": param_count, "device": str(device)})

    optimizer = make_optimizer(model, args)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.precision == "fp16"))

    rows = raw_rows(args)
    sampler = RawBatchSampler(rows, args)
    eval_sampler = RawBatchSampler(raw_rows(args), args)
    started = time.time()
    deadline = started + args.duration_hours * 3600 if args.duration_hours > 0 else math.inf
    step = 0

    for step in range(1, args.steps + 1):
        if time.time() >= deadline:
            log(out, {"event": "deadline", "step": step})
            break
        batch = sampler.next_batch(args.batch_size)
        metrics = train_step(model, tokenizer, optimizer, scheduler, scaler, batch, args, device)
        elapsed = time.time() - started
        if step == 1 or step % args.log_every == 0:
            metrics.update(
                {
                    "event": "train",
                    "step": step,
                    "elapsed_s": round(elapsed, 2),
                    "eta_h": round((args.steps - step) * (elapsed / step) / 3600, 2),
                    "lr": scheduler.get_last_lr()[0],
                }
            )
            log(out, metrics)
        if args.eval_every > 0 and step % args.eval_every == 0:
            log(out, {"event": "eval", "step": step, **evaluate(model, tokenizer, eval_sampler, args, device)})
        if args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(out, model, args, step)
        if args.dry_run_batches and step >= args.dry_run_batches:
            log(out, {"event": "dry_run_done", "step": step})
            break

    save_checkpoint(out, model, args, step)
    log(out, {"event": "done", "step": step, "elapsed_s": round(time.time() - started, 2)})


def build_model(args: Args, tokenizer: Any) -> SmallUniXcoder:
    config = small_unixcoder_config(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id or tokenizer.cls_token_id,
        eos_token_id=tokenizer.eos_token_id or tokenizer.sep_token_id,
        max_position_embeddings=args.max_len + 2,
    )
    return SmallUniXcoder(config)


class RawBatchSampler:
    def __init__(self, rows: Iterable[dict[str, str]], args: Args) -> None:
        self.rows = iter(rows)
        self.args = args
        self.rng = random.Random(args.seed)
        self.buffer: list[dict[str, str]] = []

    def next_batch(self, batch_size: int) -> dict[str, list[str]]:
        codes: list[str] = []
        docs: list[str] = []
        while len(codes) < batch_size:
            row = self._next_row()
            if not row:
                continue
            codes.append(row["code"])
            docs.append(row["doc"])
        return {"code": codes, "doc": docs}

    def _next_row(self) -> dict[str, str] | None:
        if self.args.shuffle_buffer <= 1:
            return next(self.rows)
        while len(self.buffer) < self.args.shuffle_buffer:
            try:
                self.buffer.append(next(self.rows))
            except StopIteration:
                break
        if not self.buffer:
            raise RuntimeError("raw dataset produced no usable rows")
        index = self.rng.randrange(len(self.buffer))
        row = self.buffer[index]
        try:
            self.buffer[index] = next(self.rows)
        except StopIteration:
            self.buffer.pop(index)
        return row


def raw_rows(args: Args) -> Iterator[dict[str, str]]:
    while True:
        yielded = 0
        for row in iter_once(args):
            code = str(row.get("whole_func_string") or row.get("func_code_string") or row.get("code") or "").strip()
            doc = str(row.get("func_documentation_string") or row.get("docstring") or row.get("doc") or "").strip()
            if len(code.split()) < args.min_code_tokens or len(doc.split()) < args.min_doc_tokens:
                continue
            yield {"code": code, "doc": doc}
            yielded += 1
            if args.max_rows_in_memory > 0 and yielded >= args.max_rows_in_memory:
                break
        if yielded == 0:
            raise RuntimeError("raw dataset produced no usable code/doc rows")


def iter_once(args: Args) -> Iterable[dict[str, Any]]:
    if args.data_files:
        return iter_data_files([Path(path) for path in args.data_files])
    if args.local_dataset_dir:
        dataset = load_from_disk(args.local_dataset_dir)
        split = dataset[args.split] if hasattr(dataset, "keys") and args.split in dataset else dataset
        return iter(split)
    return iter(
        load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=args.split,
            streaming=args.streaming,
        )
    )


def iter_data_files(paths: list[Path]) -> Iterator[dict[str, Any]]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.jsonl")))
            files.extend(sorted(path.rglob("*.json")))
            files.extend(sorted(path.rglob("*.parquet")))
        else:
            files.append(path)
    if not files:
        raise FileNotFoundError("no raw data files found")
    for path in files:
        if path.suffix == ".parquet":
            table = pq.read_table(path)
            for row in table.to_pylist():
                yield row
        else:
            with path.open(encoding="utf-8-sig") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)


def train_step(
    model: SmallUniXcoder,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
    batch: dict[str, list[str]],
    args: Args,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    code_inputs = tokenize(tokenizer, batch["code"], args.max_len, device)
    doc_inputs = tokenize(tokenizer, batch["doc"], args.max_len, device)
    masked_inputs, labels = mask_tokens(code_inputs["input_ids"], tokenizer, args.mlm_probability)
    code_inputs = {**code_inputs, "input_ids": masked_inputs}
    optimizer.zero_grad(set_to_none=True)
    amp_dtype = autocast_dtype(args.precision)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda" and amp_dtype is not None)):
        code_hidden = model(**code_inputs).last_hidden_state
        logits = model.lm_head(code_hidden)
        mlm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        code_vec = pool(code_hidden, code_inputs["attention_mask"])
        doc_vec = encode(model, doc_inputs)
        contrastive_loss, retrieval_acc = code_doc_contrastive(code_vec, doc_vec, args.temperature)
        loss = mlm_loss + args.contrastive_weight * contrastive_loss

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu())
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    return {
        "loss": float(loss.detach().cpu()),
        "mlm_loss": float(mlm_loss.detach().cpu()),
        "contrastive_loss": float(contrastive_loss.detach().cpu()),
        "retrieval_acc": float(retrieval_acc.detach().cpu()),
        "grad_norm": round(grad_norm, 4),
    }


@torch.no_grad()
def evaluate(
    model: SmallUniXcoder,
    tokenizer: Any,
    sampler: RawBatchSampler,
    args: Args,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    values: dict[str, list[float]] = {"loss": [], "mlm_loss": [], "contrastive_loss": [], "retrieval_acc": []}
    for _ in range(args.eval_batches):
        metrics = eval_step(model, tokenizer, sampler.next_batch(args.batch_size), args, device)
        for key in values:
            values[key].append(metrics[key])
    return {f"eval_{key}": sum(items) / max(1, len(items)) for key, items in values.items()}


@torch.no_grad()
def eval_step(
    model: SmallUniXcoder,
    tokenizer: Any,
    batch: dict[str, list[str]],
    args: Args,
    device: torch.device,
) -> dict[str, float]:
    code_inputs = tokenize(tokenizer, batch["code"], args.max_len, device)
    doc_inputs = tokenize(tokenizer, batch["doc"], args.max_len, device)
    masked_inputs, labels = mask_tokens(code_inputs["input_ids"], tokenizer, args.mlm_probability)
    code_inputs = {**code_inputs, "input_ids": masked_inputs}
    amp_dtype = autocast_dtype(args.precision)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda" and amp_dtype is not None)):
        code_hidden = model(**code_inputs).last_hidden_state
        logits = model.lm_head(code_hidden)
        mlm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        contrastive_loss, retrieval_acc = code_doc_contrastive(pool(code_hidden, code_inputs["attention_mask"]), encode(model, doc_inputs), args.temperature)
        loss = mlm_loss + args.contrastive_weight * contrastive_loss
    return {
        "loss": float(loss.cpu()),
        "mlm_loss": float(mlm_loss.cpu()),
        "contrastive_loss": float(contrastive_loss.cpu()),
        "retrieval_acc": float(retrieval_acc.cpu()),
    }


def tokenize(tokenizer: Any, texts: list[str], max_len: int, device: torch.device) -> dict[str, torch.Tensor]:
    batch = unixcoder_tokenize(
        tokenizer,
        texts,
        mode=ENCODER_ONLY,
        padding="longest",
        max_length=max_len,
        return_tensors="pt",
    )
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def mask_tokens(
    input_ids: torch.Tensor,
    tokenizer: Any,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = input_ids.clone()
    special = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in special_token_ids(tokenizer):
        special |= input_ids.eq(token_id)
    probability_matrix = torch.full(labels.shape, probability, device=input_ids.device)
    probability_matrix.masked_fill_(special, 0.0)
    masked = torch.bernoulli(probability_matrix).bool()
    labels[~masked] = -100
    if not masked.any():
        candidates = (~special).nonzero(as_tuple=False)
        if len(candidates) > 0:
            masked[tuple(candidates[0])] = True
            labels[~masked] = -100

    result = input_ids.clone()
    replace = torch.bernoulli(torch.full(labels.shape, 0.8, device=input_ids.device)).bool() & masked
    result[replace] = mask_token_id(tokenizer)
    random_replace = torch.bernoulli(torch.full(labels.shape, 0.5, device=input_ids.device)).bool() & masked & ~replace
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=input_ids.device)
    result[random_replace] = random_words[random_replace]
    return result, labels


def special_token_ids(tokenizer: Any) -> set[int]:
    ids = set(tokenizer.all_special_ids)
    ids.add(tokenizer.convert_tokens_to_ids(ENCODER_ONLY))
    return {int(item) for item in ids if item is not None and item >= 0}


def mask_token_id(tokenizer: Any) -> int:
    token_id = tokenizer.mask_token_id
    if token_id is not None:
        return int(token_id)
    token_id = tokenizer.convert_tokens_to_ids("<mask0>")
    if token_id is None or token_id < 0 or token_id == tokenizer.unk_token_id:
        raise ValueError("tokenizer has no mask token or UniXcoder <mask0> token")
    return int(token_id)


def encode(model: SmallUniXcoder, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    hidden = model(**inputs).last_hidden_state
    return pool(hidden, inputs["attention_mask"])


def pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1).div(torch.clamp(mask.sum(dim=1), min=1.0)).float()


def code_doc_contrastive(
    code_vec: torch.Tensor,
    doc_vec: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    code = F.normalize(code_vec, dim=-1)
    doc = F.normalize(doc_vec, dim=-1)
    logits = code @ doc.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    acc = (logits.argmax(dim=1) == labels).float().mean()
    return loss, acc


def save_checkpoint(out: Path, model: SmallUniXcoder, args: Args, step: int) -> None:
    payload = {
        "step": step,
        "model_name": args.model_name,
        "model_class": "SmallUniXcoder",
        "model_config": model.config.to_dict(),
        "ctx_model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "args": asdict(args),
        "pretraining_objective": "raw_codesearchnet_mlm_code_doc_contrastive",
    }
    tmp = out / "latest.pt.tmp"
    latest = out / "latest.pt"
    torch.save(payload, tmp)
    tmp.replace(latest)
    if args.save_every > 0 and step % (args.save_every * 5) == 0:
        shutil.copy2(latest, out / f"checkpoint-step-{step:08d}.pt")


def make_optimizer(model: SmallUniXcoder, args: Args) -> torch.optim.Optimizer:
    if torch.cuda.is_available():
        try:
            return AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True)
        except TypeError:
            pass
    return AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def autocast_dtype(precision: str) -> torch.dtype | None:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log(out: Path, row: dict[str, Any]) -> None:
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **row}
    with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
