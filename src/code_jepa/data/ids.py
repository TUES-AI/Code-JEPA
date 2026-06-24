"""Stable identifiers for derived code-data records."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_hash(parts: list[Any] | tuple[Any, ...], *, length: int = 20) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
