#!/usr/bin/env python3
"""Compatibility wrapper for the experimental raw-string PyTorch Siamese trainer.

The optimized path is:
1. scripts/tokenize_jepa_triples.py
2. scripts/train_siamese_bpe_jepa.py

This wrapper is kept so old commands fail less surprisingly.
"""

from __future__ import annotations

import runpy
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "experimental" / "train_siamese_bpe_jepa_torch.py"
runpy.run_path(str(SCRIPT), run_name="__main__")
