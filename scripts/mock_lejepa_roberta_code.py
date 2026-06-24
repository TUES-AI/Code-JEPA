#!/usr/bin/env python3
"""Run a local RoBERTa Code-LeJEPA mock forward pass on code text."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from code_jepa.models import RobertaCodeLeJepa


SAMPLE_CODE = """\
def contains(nums, target):
    for i in range(len(nums)):
        if nums[i] == target:
            return True
    return False
"""


SAMPLE_TARGET_CODE = """\
def contains(nums, target):
    for value in nums:
        if value == target:
            return True
    return False
"""


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    context_code = read_code(args.code_file, SAMPLE_CODE)
    target_code = read_code(args.target_code_file, SAMPLE_TARGET_CODE)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    model = RobertaCodeLeJepa.from_pretrained(
        args.model_name,
        projection_dim=args.projection_dim,
        num_slices=args.num_slices,
        sigreg_weight=args.sigreg_weight,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    context_batch = tokenize(tokenizer, context_code, args.max_length, device)
    target_batch = tokenize(tokenizer, target_code, args.max_length, device)

    with torch.no_grad():
        context = model(**context_batch)
        pair = model.forward_pair(
            context_input_ids=context_batch["input_ids"],
            context_attention_mask=context_batch["attention_mask"],
            target_input_ids=target_batch["input_ids"],
            target_attention_mask=target_batch["attention_mask"],
        )

    print("Code-LeJEPA RoBERTa mockup")
    print(f"model_name: {args.model_name}")
    print(f"context tokens: {int(context_batch['attention_mask'].sum().item())}")
    print(f"semantic head: {tuple(context.semantic.shape)}")
    print(f"local head: {tuple(context.local.shape)}")
    print(f"loss: {pair.loss.item():.6f}")
    print(f"semantic_jepa_loss: {pair.semantic_jepa_loss.item():.6f}")
    print(f"local_jepa_loss: {pair.local_jepa_loss.item():.6f}")
    print(f"sigreg_loss: {pair.sigreg_loss.item():.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="roberta-base")
    parser.add_argument("--code-file", type=Path, default=None)
    parser.add_argument("--target-code-file", type=Path, default=None)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--num-slices", type=int, default=64)
    parser.add_argument("--sigreg-weight", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def read_code(path: Path | None, fallback: str) -> str:
    if path is None:
        return fallback
    return path.read_text(encoding="utf-8")


def tokenize(tokenizer, code: str, max_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    batch = tokenizer(
        code,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {key: value.to(device) for key, value in batch.items()}


if __name__ == "__main__":
    main()
