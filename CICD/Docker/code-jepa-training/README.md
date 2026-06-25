# code-jepa-training image

GPU image for Code-JEPA data loading and model training.

Image tag pushed by CI:

```text
bonanc/code-jepa:latest
```

Quick checks inside a container:

```bash
python /usr/local/bin/verify-gpu-stack
nvidia-smi
```

Full bucket sync helper:

```bash
sync-code-jepa-all
```

No S3 startup script is fetched automatically. Start training explicitly after the pod is ready.
