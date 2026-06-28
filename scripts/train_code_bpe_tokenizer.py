#!/usr/bin/env python3
"""Train a small byte-level BPE tokenizer on prepared Code-JEPA code units."""

from __future__ import annotations

import argparse
import json
import time
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
    parser.add_argument("--max-units", type=int, default=0, help="maximum unit rows to train on; 0 means all units")
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
    iterator = CodeIterator(shards, max_units=args.max_units, batch_size=args.batch_size)
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        iterator,
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
    special_tokens_map = {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
    }
    (out / "special_tokens_map.json").write_text(json.dumps(special_tokens_map, indent=2) + "\n")

    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast.from_pretrained(str(out))
    sample = "def add(a, b):\n    return a + b\n"
    encoded = tok(sample, add_special_tokens=True)
    manifest = {
        "format": "code-jepa-byte-bpe-tokenizer-v1",
        "created_at_unix": time.time(),
        "input_roots": [str(path.expanduser().resolve()) for path in args.input_roots],
        "input_table": "units.code",
        "input_shards": len(shards),
        "trained_units": iterator.seen,
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": len(tok),
        "min_frequency": args.min_frequency,
        "max_units": None if args.max_units <= 0 else args.max_units,
        "model_max_length": args.model_max_length,
        "special_tokens": special_tokens_map,
        "special_ids": {
            "pad": tok.pad_token_id,
            "bos": tok.bos_token_id,
            "eos": tok.eos_token_id,
            "unk": tok.unk_token_id,
        },
        "sample_tokens": len(encoded["input_ids"]),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "done", "output_dir": str(out), **manifest}, sort_keys=True), flush=True)


def discover_unit_shards(roots: list[Path]) -> list[Path]:
    shards: list[Path] = []
    for root in roots:
        root = root.expanduser()
        direct = root / "units"
        if direct.exists():
            shards.extend(sorted(direct.glob("*.parquet")))
            continue
        shards.extend(sorted(root.rglob("units/*.parquet")))
    return shards


class CodeIterator:
    def __init__(self, shards: list[Path], *, max_units: int, batch_size: int) -> None:
        self.shards = shards
        self.max_units = max_units
        self.batch_size = batch_size
        self.seen = 0

    def __iter__(self) -> Iterable[str]:
        total = self.max_units if self.max_units > 0 else None
        progress = tqdm(total=total, desc="bpe-units", unit="unit")
        try:
            for shard in self.shards:
                pf = pq.ParquetFile(shard)
                for batch in pf.iter_batches(columns=["code"], batch_size=self.batch_size):
                    codes = batch.column("code").to_pylist()
                    for code in codes:
                        if not isinstance(code, str) or not code.strip():
                            continue
                        yield code
                        self.seen += 1
                        progress.update(1)
                        if self.max_units > 0 and self.seen >= self.max_units:
                            return
        finally:
            progress.close()


if __name__ == "__main__":
    main()
