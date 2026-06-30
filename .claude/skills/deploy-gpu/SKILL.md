---
name: deploy-gpu 
description: Create/start/stop/delete a RunPod GPU Pod using the existing template named `Code-JEPA`. Includes Spot (Secure Cloud) via interruptible Pods.
---

# RunPod Code-JEPA Pod

Use the helper script instead of raw `runpodctl`/`curl` commands:

```bash
.claude/skills/deploy-gpu/scripts/runpod-gpu.sh create --gpu auto --name my-job --wait
```

The script always uses the `Code-JEPA` template id `hzsxd14fdw`, keeps
everything on Secure Cloud, handles the noisy CLI/REST fallback internally, and
prints a concise summary instead of raw RunPod payloads.

**Read `gpu-remote-exec` after the pod is ready.**

## Core rules

* Secure Cloud only.
* Use only the existing template `hzsxd14fdw` (`Code-JEPA`).
* Prefer the helper script over manual commands.
* Use exact GPU if the user asks for one, otherwise use `--gpu auto`.
* `--spot` means interruptible Secure Cloud via REST.
* The repo must not be baked into the image or stored in S3.
* S3 is only for data/tokenizers/checkpoints/artifacts.

## Runtime layout

```text
/proj               root home and login/work dir
/proj/Code-JEPA     repo copied from local machine after pod startup
/proj/s3            full s3://code-jepa/ mirror, synced on startup
/proj/huggingface   HF cache
/opt/venv           image Python/JAX/PyTorch env, outside the /proj volume
```

The image starts Tailscale and runs:

```bash
sync-code-jepa-all
```

which mirrors `s3://code-jepa/` to `/proj/s3/`. Do not create `sync_dirs.txt`, S3 `entrypoint.sh`, or repo tarballs for normal deployment.

## Recommended commands

Create with fallback order:

```bash
.claude/skills/deploy-gpu/scripts/runpod-gpu.sh create --gpu auto --name my-job --wait
```

Create exact GPU:

```bash
.claude/skills/deploy-gpu/scripts/runpod-gpu.sh create --gpu "NVIDIA RTX A6000" --name my-job --wait
```

Create spot:

```bash
.claude/skills/deploy-gpu/scripts/runpod-gpu.sh create --gpu "NVIDIA RTX A40" --name my-job --spot --wait
```

List / stop / remove:

```bash
.claude/skills/deploy-gpu/scripts/runpod-gpu.sh list
.claude/skills/deploy-gpu/scripts/runpod-gpu.sh stop "$RUNPOD_POD_ID"
.claude/skills/deploy-gpu/scripts/runpod-gpu.sh remove "$RUNPOD_POD_ID"
```

If you only need the id:

```bash
.claude/skills/deploy-gpu/scripts/runpod-gpu.sh create --gpu auto --name my-job --id-only
```

## Auto GPU order

`--gpu auto` tries:

```text
NVIDIA RTX A6000 -> NVIDIA RTX A5000 -> NVIDIA RTX A4500 -> NVIDIA RTX A4000 -> NVIDIA A40 -> NVIDIA GeForce RTX 5090
```

This matches the user's preferred fallback order.

# What to do when the Pod starts

After the pod is created, check for its Tailscale host:

```bash
tailscale status | rg gpu
```

Do **not** wait blindly on Tailscale. If the host is not visible after 3-5 minutes, treat it as a startup failure and inspect the pod logs in the RunPod UI before waiting longer. The local `runpodctl` version does not expose container logs.

Copy the current local repo state from the repo root after SSH works:

```bash
ssh root@gpu-box 'rm -rf /proj/Code-JEPA && mkdir -p /proj/Code-JEPA'
git ls-files -z --cached --others --exclude-standard \
  | rsync -az --from0 --files-from=- ./ root@gpu-box:/proj/Code-JEPA/
ssh root@gpu-box 'cd /proj/Code-JEPA && /opt/venv/bin/pip install -e ".[dev,transforms]"'
```

This sends tracked files plus untracked non-ignored files, and avoids copying `.git`, `.env`, local datasets, checkpoints, and caches.

Bad startup signs to look for in logs:

```text
s3://giant-data/
/proj/giant-data
/proj/code-jepa
/opt/code-jepa-image
Code-JEPA/repo/code-jepa-*.tar.gz
sync_dirs.txt from S3
entrypoint never reaches Tailscale startup / no [code-jepa] ready line
```

If any of those appear, the pod is using an old startup/template path. Remove it immediately and fix the template/startup config before creating another pod; do not leave it retrying.

You should run commands on the remote GPU only as described in the `gpu-remote-exec` skill.

# What to do when the Pod stops / you are terminating it

When you stop or terminate the pod, push generated artifacts to S3 first, for example:

```bash
s5cmd sync --size-only /proj/s3/checkpoints/ "s3://code-jepa/checkpoints/"
```

Replace that path with the current job's checkpoint/output directory. Pods are ephemeral; upload artifacts before removal.

## Notes

* The helper script reads `RUNPOD_API_KEY` from the environment or `~/.runpod/config.toml`.
* `--wait` calls `.claude/skills/deploy-gpu/scripts/wait-new-gpu.sh` for you.
* Use spot only if the user explicitly asked for it.
