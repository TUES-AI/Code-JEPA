---
name: S3-bucket-ops
description: Use s5cmd for S3 listing, move, copy, and sync tasks.
---

# s5cmd Ops

Use this skill to interact with S3 via `s5cmd` for listing, copying, moving, and syncing artifacts.

## Core rules

* Before using `s5cmd`, ensure S3/R2 env vars are loaded from the project root: `set -a; source .env; set +a`.
* Never print secret values. Check only presence with commands like `test -n "${AWS_ACCESS_KEY_ID:-}"`.
* The project bucket is `"s3://code-jepa/"`; use it for Code-JEPA data/tokenizers/checkpoints/artifacts.
* Do not store the repo, repo tarballs, or local working-tree snapshots in S3.
* On GPU pods the bucket mirror lives under `/proj/s3`; on the Mac use `/Volumes/SSD/datasets/code-jepa` when a local mirror is needed.
* Use only `s5cmd` for S3 operations.
* Always quote S3 URIs with double quotes.
* `s5cmd mv` and `s5cmd cp` require wildcards when moving directories. Use `"s3://code-jepa/path/*"`.
* Avoid destructive deletes unless explicitly requested.
* Avoid using `s5cmd sync` to or from the S3 bucket unless necessary. If you sync, use `--size-only` because the user's S3 bucket is actually R2 and behaves badly with timestamps.

## Common commands

Load credentials from the project root if needed:

```bash
set -a; source .env; set +a
```

List buckets or prefixes:

```bash
s5cmd ls "s3://code-jepa/prefix/"
```

Move a folder (use wildcard):

```bash
s5cmd mv "s3://code-jepa/src/*" "s3://code-jepa/dst/"
```

Copy a single file:

```bash
s5cmd cp "s3://code-jepa/path/file" "/local/path/"
```

Sync the full bucket mirror on a GPU pod:

```bash
s5cmd sync --size-only "s3://code-jepa/*" "/proj/s3/"
```

Sync a prefix locally:

```bash
s5cmd sync --size-only "s3://code-jepa/prefix/" "/local/dir/"
```

## Notes

* Keep local destinations under the selected data root when syncing training artifacts.
* When unsure of a path, list parent prefixes first.
