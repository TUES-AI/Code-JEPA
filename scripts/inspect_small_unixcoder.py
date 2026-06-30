#!/usr/bin/env python3
"""Print the small UniXcoder-compatible Code-JEPA model shape and parameter count."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from code_jepa.models import SmallUniXcoder, count_parameters, small_unixcoder_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=16_384)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--save-config", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = small_unixcoder_config(
        vocab_size=args.vocab_size,
        max_position_embeddings=args.max_length + 2,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        intermediate_size=args.intermediate_size,
    )
    model = SmallUniXcoder(config)
    record = {
        "model": "small-unixcoder-code-jepa",
        "unique_parameters": count_parameters(model),
        "target_parameter_band": [25_000_000, 30_000_000],
        "config": {
            "vocab_size": config.vocab_size,
            "max_position_embeddings": config.max_position_embeddings,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "intermediate_size": config.intermediate_size,
            "is_decoder": config.is_decoder,
            "type_vocab_size": config.type_vocab_size,
        },
    }
    print(json.dumps(record, indent=2, sort_keys=True))
    if args.save_config is not None:
        args.save_config.mkdir(parents=True, exist_ok=True)
        config.to_json_file(args.save_config / "config.json")


if __name__ == "__main__":
    main()
