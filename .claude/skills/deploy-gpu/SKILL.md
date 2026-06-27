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

# How to prepare before the pod starts

1. Make sure all the needed data is in the S3 bucket
2. Create a sync_dirs.txt and entrypoint.sh which will be read on Pod startup, synced and executed. They live locally on the mac at:
```
/proj/code-jepa/sync_dirs.txt
/proj/code-jepa/entrypoint.sh
```
You will edit them and overwrite them for the current task. Preffer tmux for the entrypoint for the user to be easy to monitor.

3. Sync them to the S3 
```bash
s5cmd cp sync_dirs.txt "s3://code-jepa/sync_dirs.txt" && s5cmd cp entrypoint.sh "s3://code-jepa/entrypoint.sh"
```

Now whatever is in the S3 and stated in sync_dirs.txt will be on the pod on startup and whatever it was in entrypoint.sh will be ran so the gpu can start working right away if needed.

This workflow is for training jobs, for quick tests it is not needed to have a entrypoing.sh


# What to do when the Pod starts

After the pod is created, check for its Tailscale host:

```bash
tailscale status | rg gpu
```

Do **not** wait blindly on Tailscale. If the host is not visible after 3-5 minutes, treat it as a startup failure and inspect the pod logs in the RunPod UI before waiting longer. The local `runpodctl` version does not expose container logs.

Bad startup signs to look for in logs:

```text
s3://giant-data/
/proj/giant-data
Code-JEPA/repo/code-jepa-*.tar.gz
sync_dirs.txt from s3://giant-data/sync_dirs.txt
NoSuchKey / 404 for giant-data objects
entrypoint never reaches Tailscale startup / no [code-jepa] ready line
```

If any of those appear, the pod is using the wrong old startup/template path. Remove it immediately and fix the template/startup config before creating another pod; do not leave it retrying. The Code-JEPA path must use `s3://code-jepa/`, `/proj/code-jepa`, and the Code-JEPA image entrypoint.

You should run commands on the remote GPU only as described in the `gpu-remote-exec` skill.

# What to do when the Pod stops / you are terminating it

When you stop or terminate the pod, push the generated artifacts to S3 first:

```bash
s5cmd sync --size-only /proj/code-jepa/checkpoints/ "s3://code-jepa/checkpoints/"
```

Replace that path with the current job's checkpoint/output directory.

Pods are ephemeral; upload artifacts before removal.

## Notes
* The helper script reads `RUNPOD_API_KEY` from the environment or `~/.runpod/config.toml`.
* `--wait` calls `.claude/skills/deploy-gpu/scripts/wait-new-gpu.sh` for you.
* Use spot only if the user explicitly asked for it.
