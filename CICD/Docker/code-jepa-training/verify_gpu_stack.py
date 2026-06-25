#!/usr/bin/env python3
"""Import-time sanity check for the Code-JEPA GPU image."""

from __future__ import annotations

import json
import platform

import jax
import torch
import transformers

payload = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda_build": torch.version.cuda,
    "torch_cuda_available": torch.cuda.is_available(),
    "jax": jax.__version__,
    "jax_devices": [str(device) for device in jax.devices()],
    "transformers": transformers.__version__,
}
print(json.dumps(payload, sort_keys=True))
