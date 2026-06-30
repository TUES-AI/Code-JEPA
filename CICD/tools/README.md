# Code-JEPA CICD tools

The committed image/workflow setup is intentionally simple:

- build/push the runtime image from `CICD/Docker/code-jepa-training/`;
- publish DockerHub image `bonanc/code-jepa:latest`;
- do not bake the repo into the image;
- do not store repo tarballs or per-job startup scripts in S3.

On startup the image syncs the whole project bucket:

```text
s3://code-jepa/ -> /proj/s3/
```

Equivalent raw command:

```bash
s5cmd sync --size-only "s3://code-jepa/*" "/proj/s3/"
```

Pods are ephemeral: push checkpoints/artifacts back to `s3://code-jepa/` before deletion.
