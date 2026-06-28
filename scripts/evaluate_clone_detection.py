#!/usr/bin/env python3
"""Code-code clone/non-clone evaluation for Code-JEPA checkpoints.

The immediately runnable mode uses prepared Code-JEPA transform triples:
anchor-positive pairs are treated as clone/behavior-preserving pairs, and
anchor-negative pairs are treated as hard non-clones. This is a stress-style
clone probe, not a substitute for POJ-104 or BigCloneBench.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code_jepa.models import ENCODER_ONLY, SmallUniXcoder, ensure_unixcoder_special_tokens
from code_jepa.models import unixcoder_tokenize
from scripts.train_codebert_jepa_torch import Predictor, load_tokenizer, model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--model-name", default="microsoft/codebert-base")
    parser.add_argument("--max-examples", type=int, default=4096)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_name = checkpoint.get("model_name", args.model_name)
    tokenizer = load_tokenizer(model_name)
    if checkpoint.get("model_class") == "SmallUniXcoder":
        ensure_unixcoder_special_tokens(tokenizer)

    model = model_from_checkpoint(checkpoint, model_name).to(device)
    model.load_state_dict(checkpoint["ctx_model"])
    predictor = Predictor(
        int(model.config.hidden_size),
        float(checkpoint.get("args", {}).get("dropout", 0.1)),
    ).to(device)
    predictor.load_state_dict(checkpoint["predictor"])

    pairs = list(load_transformed_pairs(args.data_roots, args.max_examples, args.max_shards))
    if not pairs:
        raise RuntimeError(
            "no usable transformed-triple pairs found; make sure selected triples have "
            "their referenced views in the same local data root"
        )
    left = [item["left"] for item in pairs]
    right = [item["right"] for item in pairs]
    labels = np.asarray([item["label"] for item in pairs], dtype=np.int64)

    left_h = encode(model, tokenizer, left, args.max_len, args.batch_size, device)
    right_h = encode(model, tokenizer, right, args.max_len, args.batch_size, device)
    h_scores = pair_scores(left_h, right_h)
    report: dict[str, Any] = {
        "dataset": "code-jepa-transformed-triples",
        "note": "anchor-positive = clone, anchor-hard-negative = non-clone stress probe",
        "pairs": len(pairs),
        "positives": int(labels.sum()),
        "negatives": int((labels == 0).sum()),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "device": str(device),
        "mean_pool_h": binary_metrics(labels, h_scores),
    }

    with torch.inference_mode():
        left_z = F.normalize(predictor(left_h.to(device, non_blocking=True)), dim=-1).cpu()
    report["predictor_left_to_right"] = binary_metrics(labels, pair_scores(left_z, right_h))

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")


def choose_device(name: str) -> torch.device:
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def load_transformed_pairs(
    roots: list[Path],
    max_examples: int,
    max_shards: int | None,
) -> Iterable[dict[str, Any]]:
    emitted = 0
    for views_path, triples_path in discover_pairs(roots, max_shards):
        view_table = pq.read_table(views_path, columns=["view_id", "code"])
        view_by_id = {
            view_id: code
            for view_id, code in zip(
                view_table.column("view_id").to_pylist(),
                view_table.column("code").to_pylist(),
            )
            if view_id and code
        }
        triple_table = pq.read_table(
            triples_path,
            columns=["anchor_view_id", "positive_view_id", "negative_view_id"],
        )
        for anchor_id, positive_id, negative_id in zip(
            triple_table.column("anchor_view_id").to_pylist(),
            triple_table.column("positive_view_id").to_pylist(),
            triple_table.column("negative_view_id").to_pylist(),
        ):
            anchor = view_by_id.get(anchor_id)
            positive = view_by_id.get(positive_id)
            negative = view_by_id.get(negative_id)
            if not anchor or not positive or not negative:
                continue
            yield {"left": anchor, "right": positive, "label": 1}
            yield {"left": anchor, "right": negative, "label": 0}
            emitted += 2
            if emitted >= max_examples:
                return


def discover_pairs(roots: list[Path], max_shards: int | None) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for root in roots:
        triples_dir = root / "triples"
        views_dir = root / "views"
        for triples_path in sorted(triples_dir.rglob("*.parquet")):
            views_path = views_dir / triples_path.relative_to(triples_dir)
            if views_path.exists():
                pairs.append((views_path, triples_path))
    if max_shards is not None:
        return pairs[:max_shards]
    return pairs


@torch.inference_mode()
def encode(
    model: nn.Module,
    tokenizer: Any,
    texts: list[str],
    max_len: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    max_len = effective_max_len(model, max_len)
    outputs: list[torch.Tensor] = []
    unixcoder_mode = isinstance(model, SmallUniXcoder)
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        if unixcoder_mode:
            batch = unixcoder_tokenize(
                tokenizer,
                batch_texts,
                mode=ENCODER_ONLY,
                padding="longest",
                max_length=max_len,
                return_tensors="pt",
            )
        else:
            batch = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = torch.sum(hidden * mask, dim=1) / torch.clamp(torch.sum(mask, dim=1), min=1.0)
        outputs.append(F.normalize(pooled.float(), dim=-1).cpu())
    return torch.cat(outputs, dim=0)


def effective_max_len(model: nn.Module, requested: int) -> int:
    config_max = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if config_max is None:
        return requested
    return max(4, min(requested, int(config_max) - 2))


def pair_scores(left: torch.Tensor, right: torch.Tensor) -> np.ndarray:
    return torch.sum(F.normalize(left, dim=-1) * F.normalize(right, dim=-1), dim=-1).numpy()


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    best = best_f1(labels, scores)
    return {
        "average_precision": average_precision(labels, scores),
        "roc_auc": roc_auc(labels, scores),
        "best_f1": best["f1"],
        "best_threshold": best["threshold"],
        "precision_at_best_f1": best["precision"],
        "recall_at_best_f1": best["recall"],
        "accuracy_at_best_f1": best["accuracy"],
        "mean_positive_score": float(scores[labels == 1].mean()),
        "mean_negative_score": float(scores[labels == 0].mean()),
        "score_gap": float(scores[labels == 1].mean() - scores[labels == 0].mean()),
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
    best = {
        "f1": 0.0,
        "threshold": float(scores.max()) if len(scores) else 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "accuracy": 0.0,
    }
    for threshold in np.unique(scores):
        pred = scores >= threshold
        tp = float(np.sum((pred == 1) & (labels == 1)))
        fp = float(np.sum((pred == 1) & (labels == 0)))
        fn = float(np.sum((pred == 0) & (labels == 1)))
        tn = float(np.sum((pred == 0) & (labels == 0)))
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best["f1"]:
            best = {
                "f1": float(f1),
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "accuracy": float((tp + tn) / max(tp + fp + fn + tn, 1.0)),
            }
    return best


if __name__ == "__main__":
    main()
