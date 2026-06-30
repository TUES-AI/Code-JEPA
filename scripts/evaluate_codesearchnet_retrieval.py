#!/usr/bin/env python3
"""Quick CodeSearchNet NL->code retrieval eval for base vs Code-JEPA-finetuned CodeBERT.

This is a fast subset check for the task family CodeBERT was evaluated on.
It embeds docstrings and code independently, then ranks the matching function
inside the sampled candidate pool.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from transformers import AutoModel

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
    parser.add_argument("--dataset-name", default="code_search_net")
    parser.add_argument("--config", default="python")
    parser.add_argument("--split", default="test")
    parser.add_argument("--local-dataset-dir", default="data/raw/codesearchnet/python")
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len-query", type=int, default=128)
    parser.add_argument("--max-len-code", type=int, default=256)
    parser.add_argument("--model-name", default="microsoft/codebert-base")
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="CPU intra-op threads; 0 uses performance-core count when detectable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_threads(args.threads)
    device = choose_device(args.device)

    rows = load_rows(args)
    queries = [row["query"] for row in rows]
    codes = [row["code"] for row in rows]
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_name = checkpoint.get("model_name", args.model_name)
    tokenizer = load_tokenizer(model_name)
    if checkpoint.get("model_class") == "SmallUniXcoder":
        ensure_unixcoder_special_tokens(tokenizer)

    report: dict[str, Any] = {
        "dataset": f"{args.dataset_name}/{args.config}/{args.split}",
        "rows": len(rows),
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "note": (
            "CodeSearchNet NL->code retrieval subset; diagonal item is the matching "
            "docstring/function pair among the sampled pool."
        ),
    }

    if not args.skip_base:
        try:
            base = build_base_model(checkpoint, args.model_name).to(device)
            base_query = encode(
                base,
                tokenizer,
                queries,
                max_len=args.max_len_query,
                batch_size=args.batch_size,
                device=device,
            )
            base_code = encode(
                base,
                tokenizer,
                codes,
                max_len=args.max_len_code,
                batch_size=args.batch_size,
                device=device,
            )
            report["base_mean_pool"] = retrieval_metrics(base_query, base_code)
            del base, base_query, base_code
            clear_device_cache(device)
        except Exception as exc:
            report["base_mean_pool_error"] = f"{type(exc).__name__}: {exc}"

    trained = model_from_checkpoint(checkpoint, model_name).to(device)
    trained.load_state_dict(checkpoint["ctx_model"])
    trained_query = encode(
        trained,
        tokenizer,
        queries,
        max_len=args.max_len_query,
        batch_size=args.batch_size,
        device=device,
    )
    trained_code = encode(
        trained,
        tokenizer,
        codes,
        max_len=args.max_len_code,
        batch_size=args.batch_size,
        device=device,
    )
    report["trained_mean_pool_h"] = retrieval_metrics(trained_query, trained_code)

    predictor = Predictor(
        int(trained.config.hidden_size),
        float(checkpoint.get("args", {}).get("dropout", 0.1)),
    ).to(device)
    predictor.load_state_dict(checkpoint["predictor"])
    with torch.inference_mode():
        pred_query = F.normalize(
            predictor(trained_query.to(device, non_blocking=True)),
            dim=-1,
        ).cpu()
    report["trained_predictor_query_to_code"] = retrieval_metrics(pred_query, trained_code)
    report["checkpoint_step"] = int(checkpoint.get("step", -1))
    report["sample_names"] = [row["name"] for row in rows[:20]]
    print(json.dumps(report, indent=2, sort_keys=True))


def configure_threads(requested: int) -> None:
    threads = requested or detect_performance_cores() or os.cpu_count() or 1
    threads = max(1, int(threads))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, min(threads, 4)))
    except RuntimeError:
        pass


def detect_performance_cores() -> int | None:
    try:
        name = subprocess.check_output(
            ["sysctl", "-n", "hw.perflevel0.name"],
            text=True,
        ).strip()
        count = subprocess.check_output(
            ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
            text=True,
        ).strip()
        if name.lower() == "performance" and count.isdigit():
            return int(count)
    except Exception:
        return None
    return None


def choose_device(name: str) -> torch.device:
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if name == "mps" or (
        name == "auto"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def load_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    local_dir = Path(args.local_dataset_dir)
    if local_dir.exists():
        dataset = load_from_disk(str(local_dir))[args.split]
    else:
        dataset = load_dataset(args.dataset_name, args.config, split=args.split, streaming=True)
    rows: list[dict[str, str]] = []
    seen_code: set[str] = set()
    for row in dataset:
        query = (row.get("func_documentation_string") or "").strip()
        code = (row.get("whole_func_string") or row.get("func_code_string") or "").strip()
        if not query or not code:
            continue
        if len(query.split()) < 3 or len(code.split()) < 8:
            continue
        if code in seen_code:
            continue
        seen_code.add(code)
        rows.append({"query": query, "code": code, "name": str(row.get("func_name") or "")})
        if len(rows) >= args.n:
            break
    if len(rows) < 10:
        raise RuntimeError(f"only found {len(rows)} usable rows")
    return rows


def build_base_model(checkpoint: dict[str, Any], fallback_model_name: str) -> nn.Module:
    if checkpoint.get("model_class") == "SmallUniXcoder":
        return model_from_checkpoint(
            {"model_class": "SmallUniXcoder", "model_config": checkpoint["model_config"]},
            fallback_model_name,
        )
    return AutoModel.from_pretrained(fallback_model_name)


@torch.inference_mode()
def encode(
    model: nn.Module,
    tokenizer: Any,
    texts: list[str],
    *,
    max_len: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    max_len = effective_max_len(model, max_len)
    outputs: list[torch.Tensor] = []
    unixcoder_mode = isinstance(model, SmallUniXcoder)
    for start in range(0, len(texts), batch_size):
        if unixcoder_mode:
            batch = unixcoder_tokenize(
                tokenizer,
                texts[start : start + batch_size],
                mode=ENCODER_ONLY,
                padding="longest",
                max_length=max_len,
                return_tensors="pt",
            )
        else:
            batch = tokenizer(
                texts[start : start + batch_size],
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


def retrieval_metrics(query: torch.Tensor, code: torch.Tensor) -> dict[str, float]:
    sims = (query @ code.T).numpy()
    order = np.argsort(-sims, axis=1)
    ranks = np.empty(sims.shape[0], dtype=np.int64)
    for index in range(sims.shape[0]):
        ranks[index] = int(np.where(order[index] == index)[0][0]) + 1
    return {
        "n": int(len(ranks)),
        "mrr": float(np.mean(1.0 / ranks)),
        "r_at_1": float(np.mean(ranks <= 1)),
        "r_at_5": float(np.mean(ranks <= 5)),
        "r_at_10": float(np.mean(ranks <= 10)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
    }


def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
