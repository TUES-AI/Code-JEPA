# Canonical data-prep pipeline

The current data-prep path is tokenizer-agnostic. It prepares code/text records only; BPE/token caches are built later.

## Entry point

```bash
PYTHONPATH=src python scripts/prepare_data.py --help
```

Archived proof-of-learning prep scripts live in `scripts/archive/old_data_prep/`.

## Output layout

```text
<output>/
  manifest.json
  segments/
    <dataset>/
      <language>/
        transform-v0/
          core/
        transform-v1/
          core/
          only-<transform>/
        transform-v2/
          core/
          only-<transform>/
      task-semantic/
```

Each segment writes Parquet + zstd shards for:

```text
files, units, spans, views, triples, relations, semantic_pairs
```

Transform stage names are only transformation-family names. They are not data-pipeline versions. Prepared `transform-v*` segments are deltas, not cumulative copies. Training recipes are cumulative by selecting multiple segments together, e.g. `v0 + v1 + v2`.

Each delta segment may keep prior-stage support views to form triplets, but it only writes training triples involving at least one transform from that segment's own stage. Parent stage manifests make the layout appendable: later transform families can be added under `only-<transform>/` without rerunning the core segment.

## Training families

- `focal_triplet`: function/method/nested-function anchor/positive/negative.
- `context_triplet`: imports/class/sibling signatures plus the focal unit.
- `local_span_triplet`: small windows around behavior-changing spans.
- `ast_aux`: flattened AST node-type sequence side channel.
- `task_semantic_pair`: accepted solutions for the same task, including cross-language pairs when available.

## Datasets

Code datasets:

```text
codesearchnet          # full six-language CodeSearchNet: python, java, javascript, go, php, ruby
codesearchnet_python   # legacy Python-only alias
codeparrot_clean_python
```

Task datasets:

```text
humaneval
mbpp
apps
codecontests
```

Datasets are loaded through Hugging Face `datasets`; use the normal HF cache. APPS is loaded from the dataset JSONL files because the old dataset script no longer works in current `datasets`. CodeContests language ids are normalized to `python2` / `python` / `cpp` / `java`.

For persistent local cache warming:

```bash
PYTHONPATH=src python scripts/prepare_data.py \
  --output-dir /tmp/code-jepa-cache-warm \
  --download-only \
  --datasets codesearchnet codeparrot_clean_python \
  --languages all \
  --task-datasets humaneval mbpp apps codecontests
```

`codeparrot_clean_python` is large; run that only when enough local cache/disk is available.

## Performance and resume

The CLI supports per-code-dataset caps and process-parallel file/task processing:

```bash
--dataset-max-files codeparrot_clean_python=300000
```

```bash
--num-workers 8 --worker-chunksize 4 --worker-buffer-size 256
```

`--resume` reuses completed segment directories with a `manifest.json` and deletes/restarts incomplete segment directories. Completed dataset/stage segments survive interruption; the active segment is the retry unit.

Task corpora cap per-task combinatorics with:

```bash
--max-task-solutions-per-task 32 --max-semantic-pairs-per-task 256
```

The default `--max-positive-views` is `16` so stage-local hard positives have room in v1/v2 segments.

## Smoke command

```bash
PYTHONPATH=src python scripts/prepare_data.py \
  --output-dir /tmp/code-jepa-prep-smoke \
  --datasets codesearchnet \
  --languages all \
  --task-datasets humaneval mbpp \
  --transform-stages v2 \
  --splits train \
  --max-examples-per-language 10 \
  --streaming \
  --shard-size 100
```
