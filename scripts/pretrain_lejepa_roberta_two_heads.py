#!/usr/bin/env python3
"""Pretrain the RoBERTa Code-LeJEPA semantic and local heads.

The script consumes the processed Code-JEPA Parquet layout:

  data_root/
    views/**/*.parquet
    triples/**/*.parquet

Each example supplies anchor, positive, and hard-negative code views. Training
uses two heads from one shared RoBERTa encoder:

- semantic head: pooled code vector
- local head: token-level vectors

Losses:

- JEPA MSE from anchor to positive for both heads
- hard-negative margin ranking for both heads
- sliced Gaussian regularization for both heads
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from code_jepa.models import RobertaCodeLeJepa
from code_jepa.models.lejepa_roberta import flatten_masked_tokens, masked_mean_pool, masked_mse


@dataclass(frozen=True)
class TrainConfig:
    data_root: str
    output_dir: str
    model_name: str = "roberta-base"
    splits: tuple[str, ...] = ("train",)
    max_triples: int = 50000
    max_length: int = 256
    batch_size: int = 8
    steps: int = 1000
    seed: int = 0
    projection_dim: int = 256
    num_slices: int = 64
    lr: float = 2e-5
    weight_decay: float = 0.01
    margin: float = 0.2
    semantic_jepa_weight: float = 1.0
    local_jepa_weight: float = 1.0
    semantic_rank_weight: float = 0.25
    local_rank_weight: float = 0.25
    sigreg_weight: float = 0.05
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    log_every: int = 10
    save_every: int = 250
    local_files_only: bool = False
    freeze_encoder: bool = False
    amp: bool = False
    symmetric: bool = True


@dataclass(frozen=True)
class TripleExample:
    anchor_code: str
    positive_code: str
    negative_code: str
    split: str
    positive_transform: str
    negative_transform: str
    negative_type: str


class CodeTripleDataset(Dataset[TripleExample]):
    def __init__(self, examples: list[TripleExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TripleExample:
        return self.examples[index]


def main() -> None:
    cfg = parse_args()
    seed_everything(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", asdict(cfg) | {"splits": list(cfg.splits)})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.amp and device.type != "cuda":
        raise SystemExit("--amp requires a CUDA device")

    print_json(
        {
            "event": "startup",
            "device": str(device),
            "torch": torch.__version__,
            "config": asdict(cfg) | {"splits": list(cfg.splits)},
        }
    )

    examples = load_examples(Path(cfg.data_root), cfg.splits, cfg.max_triples, cfg.seed)
    if len(examples) < cfg.batch_size:
        raise SystemExit(f"Not enough examples: {len(examples)} < batch_size {cfg.batch_size}")
    print_json({"event": "loaded_examples", "count": len(examples)})

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        local_files_only=cfg.local_files_only,
    )
    model = RobertaCodeLeJepa.from_pretrained(
        cfg.model_name,
        projection_dim=cfg.projection_dim,
        num_slices=cfg.num_slices,
        sigreg_weight=cfg.sigreg_weight,
        local_files_only=cfg.local_files_only,
    ).to(device)
    if cfg.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    tokenizer.save_pretrained(output_dir / "tokenizer")
    model.encoder.config.to_json_file(output_dir / "encoder_config.json")

    dataset = CodeTripleDataset(examples)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=lambda batch: batch,
        generator=generator,
        drop_last=True,
    )

    metrics_path = output_dir / "metrics.jsonl"
    start = time.time()
    step = 0
    model.train()

    while step < cfg.steps:
        for examples_batch in loader:
            step += 1
            batch = tokenize_batch(tokenizer, examples_batch, cfg.max_length, device)

            optimizer.zero_grad(set_to_none=True)
            context = torch.autocast(device_type="cuda", dtype=torch.float16) if cfg.amp else nullcontext()
            with context:
                losses = compute_losses(model, batch, cfg)
                loss = losses["loss"]

            if cfg.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip_norm)
                optimizer.step()

            if step == 1 or step % cfg.log_every == 0:
                record = {
                    "event": "train",
                    "step": step,
                    "elapsed_s": round(time.time() - start, 2),
                    "examples_seen": step * cfg.batch_size,
                }
                record.update({name: float(value.detach().cpu()) for name, value in losses.items()})
                append_jsonl(metrics_path, record)
                print_json(record)

            if step % cfg.save_every == 0:
                save_checkpoint(output_dir, model, optimizer, scaler, cfg, step, losses)

            if step >= cfg.steps:
                break

    save_checkpoint(output_dir, model, optimizer, scaler, cfg, step, losses, final=True)
    print_json({"event": "done", "steps": step, "output_dir": str(output_dir)})


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/processed/smoke/codesearchnet-python")
    parser.add_argument("--output-dir", default="runs/lejepa-roberta-two-heads")
    parser.add_argument("--model-name", default="roberta-base")
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--max-triples", type=int, default=50000, help="Use 0 to load all triples.")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--num-slices", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--semantic-jepa-weight", type=float, default=1.0)
    parser.add_argument("--local-jepa-weight", type=float, default=1.0)
    parser.add_argument("--semantic-rank-weight", type=float, default=0.25)
    parser.add_argument("--local-rank-weight", type=float, default=0.25)
    parser.add_argument("--sigreg-weight", type=float, default=0.05)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-symmetric", dest="symmetric", action="store_false")
    args = parser.parse_args()
    return TrainConfig(**vars(args) | {"splits": tuple(args.splits)})


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_examples(
    data_root: Path,
    splits: tuple[str, ...],
    max_triples: int,
    seed: int,
) -> list[TripleExample]:
    triple_files = sorted((data_root / "triples").rglob("*.parquet"))
    view_files = sorted((data_root / "views").rglob("*.parquet"))
    if not triple_files:
        raise FileNotFoundError(f"No triple parquet files under {data_root / 'triples'}")
    if not view_files:
        raise FileNotFoundError(f"No view parquet files under {data_root / 'views'}")

    split_set = set(splits)
    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    seen = 0

    columns = [
        "split",
        "anchor_view_id",
        "positive_view_id",
        "negative_view_id",
        "positive_transform",
        "negative_transform",
        "negative_type",
    ]
    for path in triple_files:
        table = pq.read_table(path, columns=columns)
        for row in table.to_pylist():
            if row["split"] not in split_set:
                continue
            seen += 1
            if max_triples <= 0:
                sampled.append(row)
                continue
            if len(sampled) < max_triples:
                sampled.append(row)
                continue
            replace_index = rng.randrange(seen)
            if replace_index < max_triples:
                sampled[replace_index] = row

    if not sampled:
        raise ValueError(f"No triples found for splits={sorted(split_set)} under {data_root}")
    rng.shuffle(sampled)

    needed_view_ids = {
        row[key]
        for row in sampled
        for key in ("anchor_view_id", "positive_view_id", "negative_view_id")
    }
    view_code: dict[str, str] = {}
    for path in view_files:
        table = pq.read_table(path, columns=["view_id", "code"])
        for row in table.to_pylist():
            view_id = row["view_id"]
            if view_id in needed_view_ids:
                view_code[view_id] = row["code"]
        if len(view_code) == len(needed_view_ids):
            break

    examples: list[TripleExample] = []
    missing = 0
    for row in sampled:
        try:
            examples.append(
                TripleExample(
                    anchor_code=view_code[row["anchor_view_id"]],
                    positive_code=view_code[row["positive_view_id"]],
                    negative_code=view_code[row["negative_view_id"]],
                    split=row["split"],
                    positive_transform=row["positive_transform"],
                    negative_transform=row["negative_transform"],
                    negative_type=row["negative_type"],
                )
            )
        except KeyError:
            missing += 1
    if missing:
        print_json({"event": "missing_views", "count": missing})
    return examples


def tokenize_batch(
    tokenizer,
    examples: list[TripleExample],
    max_length: int,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    encoded = {}
    for field in ("anchor_code", "positive_code", "negative_code"):
        texts = [getattr(example, field) for example in examples]
        batch = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded[field.removesuffix("_code")] = {
            name: tensor.to(device) for name, tensor in batch.items() if name in {"input_ids", "attention_mask"}
        }
    return encoded


def compute_losses(
    model: RobertaCodeLeJepa,
    batch: dict[str, dict[str, torch.Tensor]],
    cfg: TrainConfig,
) -> dict[str, torch.Tensor]:
    anchor = model(**batch["anchor"])
    positive = model(**batch["positive"])
    negative = model(**batch["negative"])

    semantic_prediction = model.semantic_predictor(anchor.semantic)
    semantic_jepa = F.mse_loss(semantic_prediction, positive.semantic.detach())

    local_prediction = model.local_predictor(anchor.local)
    local_jepa = local_prediction_loss(
        local_prediction,
        anchor.attention_mask,
        positive.local.detach(),
        positive.attention_mask,
    )

    if cfg.symmetric:
        reverse_semantic_prediction = model.semantic_predictor(positive.semantic)
        semantic_jepa = 0.5 * (
            semantic_jepa + F.mse_loss(reverse_semantic_prediction, anchor.semantic.detach())
        )
        reverse_local_prediction = model.local_predictor(positive.local)
        local_jepa = 0.5 * (
            local_jepa
            + local_prediction_loss(
                reverse_local_prediction,
                positive.attention_mask,
                anchor.local.detach(),
                anchor.attention_mask,
            )
        )

    semantic_rank = margin_rank_loss(
        semantic_prediction,
        positive.semantic.detach(),
        negative.semantic.detach(),
        cfg.margin,
    )
    local_rank = margin_rank_loss(
        pooled_local(local_prediction, anchor.attention_mask),
        pooled_local(positive.local.detach(), positive.attention_mask),
        pooled_local(negative.local.detach(), negative.attention_mask),
        cfg.margin,
    )
    sigreg = model.semantic_sigreg(
        torch.cat([anchor.semantic, positive.semantic, negative.semantic], dim=0)
    ) + model.local_sigreg(
        torch.cat(
            [
                flatten_masked_tokens(anchor.local, anchor.attention_mask),
                flatten_masked_tokens(positive.local, positive.attention_mask),
                flatten_masked_tokens(negative.local, negative.attention_mask),
            ],
            dim=0,
        )
    )

    loss = (
        cfg.semantic_jepa_weight * semantic_jepa
        + cfg.local_jepa_weight * local_jepa
        + cfg.semantic_rank_weight * semantic_rank
        + cfg.local_rank_weight * local_rank
        + cfg.sigreg_weight * sigreg
    )
    return {
        "loss": loss,
        "semantic_jepa_loss": semantic_jepa,
        "local_jepa_loss": local_jepa,
        "semantic_rank_loss": semantic_rank,
        "local_rank_loss": local_rank,
        "sigreg_loss": sigreg,
    }


def local_prediction_loss(
    prediction: torch.Tensor,
    prediction_mask: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    seq_len = min(prediction.size(1), target.size(1))
    mask = prediction_mask[:, :seq_len] * target_mask[:, :seq_len]
    return masked_mse(prediction[:, :seq_len], target[:, :seq_len], mask)


def pooled_local(local: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    return F.normalize(masked_mean_pool(local, attention_mask), dim=-1)


def margin_rank_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negative = F.normalize(negative, dim=-1)
    pos_sim = F.cosine_similarity(anchor, positive, dim=-1)
    neg_sim = F.cosine_similarity(anchor, negative, dim=-1)
    return F.relu(margin + neg_sim - pos_sim).mean()


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cfg: TrainConfig,
    step: int,
    losses: dict[str, torch.Tensor],
    *,
    final: bool = False,
) -> None:
    name = "checkpoint-final.pt" if final else f"checkpoint-step-{step:06d}.pt"
    path = output_dir / name
    torch.save(
        {
            "step": step,
            "config": asdict(cfg) | {"splits": list(cfg.splits)},
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "losses": {name: float(value.detach().cpu()) for name, value in losses.items()},
        },
        path,
    )
    print_json({"event": "checkpoint", "step": step, "path": str(path), "final": final})


def print_json(record: dict[str, Any]) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
