# code-jepa-training image

GPU runtime image for Code-JEPA data loading and model training.

Image tag pushed by CI:

```text
bonanc/code-jepa:latest
```

Quick checks inside a container:

```bash
python /usr/local/bin/verify-gpu-stack
nvidia-smi
```

Startup behavior:

```text
s3://code-jepa/ -> /proj/s3/
```

The image does not contain the repo. `/opt/venv` is outside the `/proj` volume and is activated for shells. After the pod is ready, sync the local working tree to `/proj/Code-JEPA` and run from there.
