#!/usr/bin/env python3
"""Train a small byte-level BPE tokenizer on prepared Code-JEPA code units."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=16_384)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-units", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--model-max-length", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    shards = discover_unit_shards(args.input_roots)
    if not shards:
        raise FileNotFoundError(f"no units shards under {args.input_roots}")

    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        code_iterator(shards, max_units=args.max_units, batch_size=args.batch_size),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens,
    )
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> $B:1 <eos>:1",
        special_tokens=[("<bos>", 1), ("<eos>", 2)],
    )
    tokenizer.save_model(str(out))
    tokenizer.save(str(out / "tokenizer.json"))

    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "model_max_length": args.model_max_length,
        "clean_up_tokenization_spaces": False,
    }
    (out / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2) + "\n")
    (out / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            },
            indent=2,
        )
        + "\n"
    )

    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast.from_pretrained(str(out))
    sample = "def add(a, b):\n    return a + b\n"
    encoded = tok(sample, add_special_tokens=True)
    print(
        json.dumps(
            {
                "event": "done",
                "output_dir": str(out),
                "vocab_size": len(tok),
                "sample_tokens": len(encoded["input_ids"]),
                "special_ids": {
                    "pad": tok.pad_token_id,
                    "bos": tok.bos_token_id,
                    "eos": tok.eos_token_id,
                    "unk": tok.unk_token_id,
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )


def discover_unit_shards(roots: list[Path]) -> list[Path]:
    shards: list[Path] = []
    for root in roots:
        shards.extend(sorted((root.expanduser() / "units").glob("*.parquet")))
    return shards


def code_iterator(shards: list[Path], *, max_units: int, batch_size: int) -> Iterable[str]:
    seen = 0
    progress = tqdm(total=max_units, desc="bpe-units", unit="unit")
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(columns=["code"], batch_size=batch_size):
            codes = batch.column("code").to_pylist()
            for code in codes:
                if isinstance(code, str) and code.strip():
                    yield code
                    seen += 1
                    progress.update(1)
                    if seen >= max_units:
                        progress.close()
                        return
    progress.close()


if __name__ == "__main__":
    main()
