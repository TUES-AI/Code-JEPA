# Code-JEPA CICD tools

The committed image/workflow setup is intentionally simple:

- build/push image from `CICD/Docker/code-jepa-training/`;
- publish DockerHub image `bonanc/code-jepa:latest`;
- do not pull an S3 startup script in the image;
- do not use per-job `sync_dirs.txt` lists.

For data on a GPU pod, sync the whole project bucket:

```bash
set -a; source .env; set +a
sync-code-jepa-all
```

Equivalent raw command:

```bash
s5cmd sync --size-only "s3://code-jepa/*" "/proj/code-jepa/"
```

Pods are still ephemeral: push checkpoints/artifacts back to `s3://code-jepa/` before deletion.
