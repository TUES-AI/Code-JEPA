# Code-JEPA Docker image

Build context for the Code-JEPA GPU training image.

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
/proj/Code-JEPA      repo workdir
/proj/code-jepa      synced project data/artifacts
/proj/huggingface    HF cache
/proj/checkpoints    training checkpoints
/proj/artifacts      run artifacts
```

No startup script is pulled from S3 by the image. If a pod needs data, sync the whole project bucket explicitly:

```bash
set -a; source .env; set +a
sync-code-jepa-all
```

That syncs:

```text
s3://code-jepa/ -> /proj/code-jepa/
```
