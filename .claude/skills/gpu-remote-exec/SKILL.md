---
name: gpu-remote-exec
description: Run commands or JAX scripts on the remote CUDA GPU box via ssh root@gpu-box when execution needs a GPU/CUDA backend or the user asks to run on the GPU box/remote container.
---

# GPU Remote Exec

Use this skill to execute commands on the CUDA-capable GPU box at `root@gpu-box`.

## Core rules

* **Edit code only locally. Never edit files on the remote GPU box.**
* After **any** local code change, re-sync the local working tree to `/proj/Code-JEPA` and then run remotely.
* Do not use S3 for repo transfer. S3 is only for data/tokenizers/checkpoints/artifacts.
* If the remote machine needs tiny data files, copy them with `scp`; for larger files or datasets use `S3-bucket-ops`.
* Never default to RunPod CLI access for commands. Use SSH as described here.

## Workflow

0. Decide if the task requires GPU/CUDA or the user explicitly asks for the GPU box.

1. Check which remote GPU answers and use it until it is down:

```bash
ssh root@gpu-box "echo ok"
ssh root@gpu-box-1 "echo ok"
ssh root@gpu-box-2 "echo ok"
```

Or inspect Tailscale:

```bash
tailscale status | rg gpu
```

2. Sync the local repo state from the repo root. This sends tracked files plus untracked non-ignored files, and excludes `.git`, `.env`, local datasets, checkpoints, and caches:

```bash
ssh root@gpu-box 'rm -rf /proj/Code-JEPA && mkdir -p /proj/Code-JEPA'
git ls-files -z --cached --others --exclude-standard \
  | rsync -az --from0 --files-from=- ./ root@gpu-box:/proj/Code-JEPA/
```

After the first sync on a fresh pod, or whenever dependencies change, install the local package:

```bash
ssh root@gpu-box 'cd /proj/Code-JEPA && /opt/venv/bin/pip install -e ".[dev,transforms]"'
```

3. Run the desired command or JAX script remotely:

```bash
ssh root@gpu-box 'set -e; cd /proj/Code-JEPA; /opt/venv/bin/python <SCRIPT_OR_MODULE> <ARGS>'
```

4. If results require code changes:

   * Make the change locally.
   * Repeat Step 2.
   * Repeat Step 3.

5. Clean up the remote box if needed:

```bash
ssh root@gpu-box 'rm -rf /proj/Code-JEPA'
```

## Useful tips

This skill is generally used with the `S3-bucket-ops` skill. On current pods, the image mirrors `s3://code-jepa/` to `/proj/s3/` at startup.

## Notes

* The remote is ephemeral; local files are the source of truth.
* If a synced command cannot import `code_jepa`, re-run the editable install command from Step 2.
* When parsing logs that include tqdm output, normalize carriage returns (`\r`) to newlines so `[eval]` lines can be found reliably.
