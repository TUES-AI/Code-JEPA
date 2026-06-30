# Code-JEPA Docker image

Build context for the Code-JEPA GPU training runtime image.

Canonical image:

```text
bonanc/code-jepa
```

Main Dockerfile:

```text
CICD/Docker/code-jepa-training/Dockerfile
```

Stack:

- Ubuntu 22.04
- CUDA 12.8 + cuDNN devel image
- Python 3.11 virtualenv at `/opt/venv`
- PyTorch CUDA 12.8 wheels
- JAX CUDA 12 wheels
- Transformers / Datasets / Tokenizers / Flax / Optax
- `s5cmd` for Cloudflare R2/S3
- Tailscale installed for remote pod access

Runtime paths:

```text
/proj               root home and login/work dir
/proj/Code-JEPA     repo copied from local machine after pod startup
/proj/s3            full s3://code-jepa/ mirror, synced on startup
/proj/huggingface   HF cache
/opt/venv           image Python/JAX/PyTorch env, outside the /proj volume
```

The repo is not baked into the image and is not pulled from S3. Copy the local working tree to `/proj/Code-JEPA` after the pod is reachable.
