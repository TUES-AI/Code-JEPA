# RoBERTa-Only Code-JEPA Ablation and Benchmark Plan

## Summary

This study uses one RoBERTa-style backbone family for all Code-JEPA ablations. Every
Code-JEPA variant should use the same bidirectional encoder shape, tokenizer,
sequence length, training budget, and downstream fine-tuning protocol unless the
variant explicitly ablates representation or pooling.

Compute is limited. The study should run at most two full pretrainings. All
screening ablations should use a smaller 25-30M parameter RoBERTa-style model and
a reduced data budget before any result is promoted to full scale.

CodeBERT, GraphCodeBERT, TreeBERT, and UniXcoder are comparison points from prior
papers or separately reproduced baselines. They are not ablation backbones.

Core claim:

> With the same RoBERTa backbone, Code-JEPA's gains come from
> behavior-preserving positives, behavior-changing hard negatives, JEPA/ranking
> losses, and projection/local representation choices.

Venue framing for REALM:

> Position Code-JEPA as a compact representation and memory model for coding
> agents: it helps rank, cluster, and avoid repeated failed code edits without
> executing every candidate. The study should emphasize data design, coding-agent
> evaluation, and robustness rather than broad foundation-model pretraining.

## Default Model

- RoBERTa-style bidirectional encoder.
- Mask-aware mean pooling over non-pad tokens.
- Projection head:

```text
Dense(8H) -> split 4H/4H -> SwiGLU -> RMSNorm -> Dense(D)
```

- Train JEPA, ranking, in-batch contrastive, and SIGReg losses on projected
  vectors `z`.
- Keep pooled encoder vectors `h` as a downstream/search embedding candidate.
- Downstream retrieval must compare both `h` and `z`.

## Study Phases

### 1. Pretraining

Match UniXcoder at the corpus/stage level, not at the architecture/objective
level. UniXcoder pretrains on C4, then CodeSearchNet unimodal code, then
CodeSearchNet code-comment-AST multimodal data. Code-JEPA should use the same
CodeSearchNet language scope where feasible, but replace UniXcoder's generation
and AST-token objectives with RoBERTa encoder training plus Code-JEPA
positive/hard-negative representation learning.

Use two pretraining tiers:

- **Small ablation tier:** 25-30M parameter RoBERTa-style encoder, fixed token
  budget, and a representative subset of `v0`/`v1`.
- **Full tier:** at most two full pretrainings, reserved for the strongest
  Code-JEPA recipe and one essential control.

Every run inside a tier should use the same RoBERTa configuration unless the
ablation explicitly changes representation or pooling.

Report these pretraining diagnostics:

```text
rank_acc
sim_pos
sim_neg
sim_gap
jepa_loss / pos_loss
rank_loss
inbatch_loss
sigreg_loss
examples_per_s
tokens_per_s
```

Small-tier pretraining comparisons to record:

```text
RoBERTa + positives only
RoBERTa + positives + random negatives
RoBERTa + positives + v0 hard negatives
RoBERTa + positives + v0+v1 hard negatives
```

Important control:

```text
Same RoBERTa architecture + same token budget + same data budget
```

This keeps the ablation from becoming a backbone comparison.

Full-tier pretraining budget:

```text
Full pretrain 1: matched RoBERTa control on the same CodeSearchNet language/data mix, without Code-JEPA hard-negative objective
Full pretrain 2: matched RoBERTa Code-JEPA on the same mix, with Code-JEPA positives/hard negatives where transform coverage exists
```

### 2. Fine-Tuning and Evaluation

Use the same pretrained checkpoint interface for every ablation.

Code search:

- Dataset: CodeSearchNet Python first.
- Later optional datasets: AdvTest and CosQA.
- Metrics:
  - MRR
  - R@1
  - R@5
  - R@10
  - median rank

Clone detection:

- Dataset: POJ-104 first if available.
- Later optional dataset: BigCloneBench.
- Metrics:
  - MAP@R or MAP for retrieval-style clone detection
  - F1 / precision / recall for pair-classification setup

Stress evaluation:

- Use held-out transformed triples.
- Split by repository/source/task before transformation so transformed views do
  not cross train/eval boundaries.
- Metrics:
  - hard-negative accuracy
  - positive similarity
  - negative similarity
  - similarity gap
  - per-negative-type accuracy

Stress categories:

```text
rename / formatting / docstring removal
comparison flip
boolean flip
wrong variable
swapped arguments
small integer / off-by-one
return removal
await removal
sort/default/subscript mutations
```

## Language Scope

Benchmark languages:

| Benchmark | Language scope |
|---|---|
| POJ-104 | C/C++ |
| BigCloneBench | Java |
| CoSQA | Python |
| AdvTest | Python |
| CSN / CodeSearchNet | Go, Java, JavaScript, PHP, Python, Ruby |

UniXcoder's pretraining language scope is CodeSearchNet's six languages:

```text
Ruby, Java, Python, PHP, Go, JavaScript
```

Code-JEPA should match that scope for the two full pretrainings if feasible.
However, current high-confidence Code-JEPA AST transforms and hard negatives are
Python-first. Therefore:

- Small-tier ablations stay Python-first to cheaply test the objective.
- Full-tier control uses the same balanced CodeSearchNet six-language corpus.
- Full-tier Code-JEPA uses the same balanced CodeSearchNet six-language corpus,
  with Code-JEPA triples for Python and any other languages once equivalent
  transform coverage exists.
- For non-Python CodeSearchNet languages without hard-negative transforms, keep
  code/doc metadata and optional dropout/in-batch contrastive examples, but do
  not claim behavior-sensitive hard-negative supervision for those languages.

This is the fairest compute-limited match: the corpus matches UniXcoder, while
the Code-JEPA signal remains explicit about where augmentations are available.

Do not pretrain directly on POJ-104 or BigCloneBench test distributions just to
cover C/C++ or Java clone detection. Treat those as downstream benchmarks.

## Matched UniXcoder-To-Code-JEPA Pretraining Schedule

Use the following conceptual match:

| UniXcoder stage | UniXcoder objective | Code-JEPA matched stage | Code-JEPA objective |
|---|---|---|---|
| C4 text warmup | MLM / ULM / denoising | Optional inherited initialization or short RoBERTa MLM warmup | MLM only if needed for a control; not central |
| CodeSearchNet unimodal code | MLM / ULM / denoising over code | CodeSearchNet six-language code control | RoBERTa encoder MLM or contrastive control on same token budget |
| CodeSearchNet code-comment-AST | contrastive + comment generation + AST input | CodeSearchNet doc/AST metadata + Code-JEPA triples | JEPA prediction, positive alignment, hard-negative ranking, in-batch contrastive, SIGReg |

Because our model is encoder-only RoBERTa-style, do not include UniXcoder's
decoder-only ULM or encoder-decoder denoising/comment-generation objectives in
the main Code-JEPA run. Those would change the model class and exceed the compute
budget.

The full Code-JEPA run should consume the same broad inputs as the control:

```text
CodeSearchNet six-language code
+ documentation/comment metadata where available
+ AST/span/sketch metadata where available
+ Code-JEPA positives and hard negatives where transform support exists
```

The full control run should consume the same code/doc/AST metadata but not use
Code-JEPA hard-negative triples. This isolates the value of the Code-JEPA
augmentation/objective rather than the value of more data.

## Data-Prep Comparison To UniXcoder

UniXcoder uses multi-modal code preparation: code comments, flattened AST tokens,
CodeSearchNet functions across six languages, tree-sitter parsing, a 50K BPE
vocabulary plus AST non-terminal special tokens, and ablations for contrastive
learning, cross-modal generation, comments, AST, and AST traversal choices.

Code-JEPA should not replicate that full setup because compute is limited and the
paper target is agent-oriented. Instead, take three low-cost lessons:

- **Keep comments/docs as metadata.** Do not only remove docstrings as positives.
  Also preserve original documentation fields for code-search fine-tuning and
  for analysis of whether comment-rich examples behave differently.
- **Add an AST-view stress slice, not a full AST pretraining system.** Use the
  existing Python AST spans and optionally a flattened AST text view for
  evaluation or a small-tier data-prep ablation. Do not add thousands of AST
  special tokens or full AST traversal pretraining in the first submission.
- **Report parse/transform coverage.** Track how many units have parseable AST,
  comments/docstrings, generated positives, generated hard negatives, and each
  negative type. This turns data prep into an explicit contribution instead of
  only an implementation detail.

Small optional data-prep ablation, only if the four required small-tier runs are
not enough:

| Variant | Purpose |
|---|---|
| v0+v1 hard negatives + doc/comment metadata filter | Tests whether comment/doc-rich examples improve code search without changing the core objective |
| v0+v1 hard negatives + AST-view stress eval | Tests whether Code-JEPA handles syntax-structure perturbations better than text-only stress tests |

Do not add multilingual training, cross-modal generation, AST special-token
vocabulary, or BFS/DFS AST traversal ablations for this submission. Those are
too expensive and would distract from the Code-JEPA agent-memory claim.

## Adapted Hybrid Data Recipe

Use a "UniXcoder-lite" data recipe that keeps Code-JEPA's hard-negative triples
as the core signal, while borrowing UniXcoder's useful data modalities in cheap
ways.

### Keep Code-JEPA Core

- Continue to train on anchor / positive / hard-negative code triples.
- Keep behavior-preserving positives and behavior-changing negatives as the main
  representation signal.
- Keep changed spans and negative type metadata as first-class evaluation fields.

### Add Lightweight Comment/Doc Support

- Preserve CodeSearchNet documentation strings as query-code metadata for code
  search fine-tuning and evaluation.
- For CodeParrot/function units, record whether a unit has a docstring, comment
  density if cheap to extract, and docstring length/availability.
- Do not make comments the main pretraining modality in this submission.
- Do not treat comment rewrites as behavior-preserving positives unless the
  transform is explicitly metadata-only; comments are useful for retrieval but
  not reliable evidence of program behavior.

### Add Lightweight AST Support

- Keep current AST spans.
- Add, when cheap, a bounded AST sketch per unit:

```text
node_type sequence
top-level/function signature shape
changed node type around hard-negative span
```

- Use AST sketches for analysis and stress slicing, not as mandatory model input.
- Do not add AST special tokens or full AST sequence pretraining for the first
  submission.

### Add Coverage Reporting

Every prepared dataset/run should report:

```text
parse_ok units
compile_ok units
units with docstrings/comments
units with at least one positive view
units with at least one hard negative
counts by positive transform
counts by negative transform/type
changed-span coverage by negative type
AST-span coverage
length buckets
```

This combines UniXcoder's strength in data accounting with Code-JEPA's targeted
hard-negative construction.

### Final Data Recipe For The Main Run

The main Code-JEPA run should use:

```text
CodeSearchNet six-language code units
+ preserved documentation/comment metadata
+ AST spans/sketches for analysis
+ v0 positives where supported
+ v0+v1 hard negatives where supported
+ transform/coverage statistics
```

This is the "best of both" setup: UniXcoder-style metadata awareness and
benchmark compatibility, but Code-JEPA-style behavior-sensitive training.

## Required Code-JEPA Ablations

Run only this reduced set first. These are small-tier ablations unless explicitly
promoted to the full tier.

| Group | Variant | Purpose |
|---|---|---|
| Data | positives only | Tests whether invariance alone explains the gains |
| Data | positives + random negatives | Tests whether generic contrastive training is enough |
| Data | positives + v0 hard negatives | Tests the first conservative behavior-changing mutation set |
| Data | positives + v0+v1 hard negatives + doc/AST metadata | Tests the final hybrid recipe and is the candidate full Code-JEPA run |

Do not run loss, pooling, sampling, or individual transform-removal ablations in
the first pass. They are deferred unless the four small-tier runs are ambiguous.
The only representation comparison kept in scope is evaluation-time use of pooled
`h` versus projected `z` from the same checkpoints, because it does not require
extra pretraining.

For a REALM submission, the highest-priority result is not matching UniXcoder on
all standard tasks. The highest-priority result is showing that a small RoBERTa
Code-JEPA representation improves coding-agent-relevant ranking and robustness:

```text
same task/reference anchor -> rank useful candidate higher
known failed attempt       -> identify duplicate failure families
harmless rewrite           -> stay close
tiny behavior bug          -> move away
```

## Result Tables

### Main Pretraining Ablation Table

```text
Variant                         Rank Acc  Sim Pos  Sim Neg  Sim Gap  Loss
positives only
positives + random negatives
positives + v0 negatives
positives + v0+v1 negatives + doc/AST metadata
full-tier control
full-tier Code-JEPA
```

### Downstream Fine-Tuning Table

```text
Variant                         CodeSearch MRR  R@1  R@5  Clone MAP/F1
RoBERTa continued pretrain control
Full RoBERTa Code-JEPA
positives only
random negatives
v0 hard negatives
v0+v1 hard negatives + doc/AST metadata
```

### Representation-Use Table

```text
Checkpoint                      Eval Embedding  CodeSearch MRR  Clone MAP/F1  Stress Acc
Full Code-JEPA                  pooled h
Full Code-JEPA                  projected z
Small-tier v0+v1                pooled h
Small-tier v0+v1                projected z
```

### Comparison-To-Paper Table

```text
Model / Data Prep               CodeSearch MRR  Clone Metric  Source
CodeBERT                        prior/reproduced baseline
GraphCodeBERT                   prior/reproduced baseline
TreeBERT                        prior/reproduced baseline
UniXcoder                       prior/reproduced baseline; comment + AST multi-modal prep
RoBERTa Code-JEPA               ours; doc/AST metadata + transform triples + hard-negative stress prep
```

This table is for context only. It must not be mixed with the RoBERTa-only
ablation table.

### Agent-Relevance Table

```text
Variant                         Rewrite Inv  HardNeg Acc  Failure-Duplicate Acc  Candidate Rerank MRR
RoBERTa continued pretrain control
Small-tier v0+v1 Code-JEPA
Full RoBERTa Code-JEPA
```

If time is tight, this table is more important for REALM than adding more
traditional ablations.

## Assumptions

- Main ablations use only RoBERTa-style encoders.
- Small-tier ablations use a 25-30M parameter RoBERTa-style model.
- The study can afford at most two full pretrainings: one control and one final
  Code-JEPA recipe.
- CodeBERT, GraphCodeBERT, TreeBERT, and UniXcoder are comparison baselines,
  not Code-JEPA ablation backbones.
- The first runnable downstream benchmark is CodeSearchNet Python because this
  repo already has partial support, but the full matched pretraining target is
  balanced CodeSearchNet across Ruby, Java, Python, PHP, Go, and JavaScript.
- POJ-104 and BigCloneBench require loaders/evaluators before clone-detection
  results can be filled.
- Non-Python hard-negative claims require non-Python transform coverage; until
  then, hard-negative supervision is Python-only and multilingual data contributes
  broad code/doc representation learning.
- Synthetic `rank_acc` is a pretraining diagnostic, not the final proof of
  semantic quality.
- REALM framing should foreground coding-agent use: candidate reranking,
  duplicate failure detection, data/simulation design, and robustness.
