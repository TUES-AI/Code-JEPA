# Project: Code-JEPA

This document is the scientific source of truth for the project. Code, experiments, and paper drafts should follow this document. Update it when the research understanding changes, not when implementation details move around.

See also: [`docs/diagrams/code-jepa-project-map.pdf`](docs/diagrams/code-jepa-project-map.pdf).

## Main idea

Train a code-native Joint-Embedding Predictive Architecture (Code-JEPA) that learns code-space geometry useful for code agents and code LLM post-training.

The representation should be:

- invariant to harmless program changes;
- sensitive to tiny behavior-changing edits;
- usable without executing code;
- useful as a frozen judge, memory, and ranker for downstream training.

The downstream name we have been using is **RLJF: Reinforcement Learning from JEPA Feedback**. The broad paper is bigger than only RLJF: it also includes hindsight feedback and failure-memory exploration for code agents.

## Core hypothesis

A Code-JEPA trained with positive behavior-preserving transformations and hard negative behavior-impacting mutations can learn a more useful code similarity space than generic code embeddings or token-level similarity.

Target geometry:

```text
renaming / formatting / safe refactor        -> close
< vs <= / wrong variable / swapped args      -> far in semantic/local space
same task, different algorithm, both correct -> semantically close, strategically maybe far
unrelated code                               -> far
```

The model is not a correctness oracle. It ranks, clusters, detects duplicate failures, and supplies preference signals when there is an anchor such as a reference solution, verifier success, accepted patch, or hindsight final solution.

## Model shape

Use a shared-encoder Siamese setup as the default paper path:

```text
code view -> shared transformer encoder -> hidden states H
          -> mask-aware mean pool        -> h
          -> projection head p(h)        -> z

anchor/positive/negative views -> same shared encoder/projection -> z_anchor/z_pos/z_neg
train z_anchor close to z_pos and far from z_neg
SIGReg(z outputs)
```

Use one shared encoder for all views and all languages: no target encoder, no EMA, and no stop-grad. Do not use an unconditioned predictor as the default; `z_anchor -> predictor -> z_pos` is only justified as an ablation unless the predictor is conditioned on transform type, language, mask/context, or target-view metadata. Apply SIGReg in the projected training space `z`, not directly to the reusable encoder embedding `h`. Use a small SIGReg coefficient around `0.03`-`0.05`; after reading LeJEPA, `0.10` should be treated as high/likely too much, not a normal setting.

The default global projection head should be:

```text
p(h) = Dense(8H) -> split 4H/4H -> SwiGLU -> RMSNorm -> Dense(D)
```

Keep `h = pool(H)` as the frozen downstream/search embedding candidate, and train JEPA/ranking/SIGReg losses on `z = p(h)`. This gives SIGReg a loss space to shape without forcing the encoder output itself to become exactly Gaussian. The current no-head pooled-output path is only a baseline and throughput path. See [`docs/design-notes/embedding-pooling-and-projection-head.md`](docs/design-notes/embedding-pooling-and-projection-head.md).

On top of the shared code encoder, use small projection/readout heads, not separate full transformers:

```text
code -> shared transformer encoder -> hidden states H
  -> strategy head  g_s(pool(H)) -> z_strategy
  -> semantic head  g_m(pool(H)) -> z_semantic
  -> local head     g_l(H)       -> token/AST/span vectors
```

The same code pair can be positive for one head and negative for another. Example: `i < n` vs `i <= n` should be strategy-close but semantic/local-far.

## Data object hierarchy

Do not flatten everything into isolated snippets. Keep whole-file context and derive trainable units from it:

```text
repo
  -> file
     -> imports / constants / top-level definitions
     -> class
        -> method
     -> function
     -> AST/local spans
```

Whole files are the canonical source and context. Functions/methods are the main training units. Local AST/token spans are needed for tiny semantic changes.

Recommended records:

- `files`: repo/path/license/source/imports/top-level defs/parse status;
- `units`: function/method/nested function/class summary/file window/span window/task solution with parent file, line range, context, identifiers, calls, and AST node-type sequence;
- `spans`: AST node spans and changed spans;
- `views`: anchor/positive/negative/context/local-span/task-solution transformed code;
- `relations`: file contains unit, class contains method, unit has AST auxiliary view, unit uses import, function calls function;
- `triples`: anchor-positive-negative training relations with per-head labels;
- `semantic_pairs`: same-task accepted-solution pairs, including same-language different implementations and cross-language solutions.

The canonical data-prep pipeline is tokenizer-agnostic and writes reproducible Parquet + zstd segments by dataset and transformation stage. Keep raw code, spans, context, AST side channels, transform metadata, rough length buckets, repo/task split, and sampling weights. Do not tokenize until the model/backbone tokenizer is chosen.

## Code block sizes

Prepare buckets instead of one global cutoff:

| Bucket | Rough size | Use |
|---|---:|---|
| tiny | 3-10 LOC / 32-128 tokens | local operator sensitivity |
| short | 10-40 LOC / 128-512 tokens | main hard-negative training |
| medium | 40-120 LOC / 512-1536 tokens | realistic function embeddings |
| long | 120-250 LOC / 1536-4096 tokens | later long-context evaluation |
| file/class | 250+ LOC | context/file JEPA, not single-vector semantic sensitivity |

First serious training should focus on 10-120 LOC functions/methods while retaining file context.

## Transformations

### Positive transformations

Positive views should preserve behavior with high confidence:

- variable/argument renaming with references updated;
- local helper renaming when safe;
- formatting, whitespace, quote normalization;
- comment/docstring removal or rewriting when safe;
- trivial import split/merge/reorder when no side-effect risk;
- equivalent syntax rewrites only under conservative rules;
- conservative control-flow rewrites such as boolean-return simplification and if-return conditional expressions;
- guarded independent-statement reordering;
- guarded refactor-like rewrites such as simple loop-to-comprehension, range-for-to-while, and accumulator-to-`sum` rewrites;
- alternate structural views: AST, DFG, CFG, call graph, dependency graph, call-site context.

Riskier positives such as broad statement reordering, algebraic rewrites, top-level definition reordering, import style normalization, type annotation changes, and function extraction/inlining should be delayed or marked with confidence flags.

### Hard negatives

Hard negatives are compile-valid behavior-impacting mutations relative to the original, not guaranteed test failures:

- `<` <-> `<=`, `>` <-> `>=`, `==` <-> `!=`;
- `+1` <-> `-1`, loop-bound changes, off-by-one edits;
- `and` <-> `or`;
- wrong variable from the same scope;
- swapped call arguments;
- wrong default value or wrong API order;
- missing return / missing await;
- wrong sort direction;
- mutate copy vs original;
- missing edge-case branch or wrong exception handling.

Always record changed byte/AST spans. The local head depends on this metadata.

### Dataset transform stages

These names describe transformation families only, not data-pipeline versions. The canonical prep pipeline emits each stage as a separate reproducible delta segment to avoid duplicating training triples when the whole dataset is processed. Training recipes are cumulative by selecting multiple segments, e.g. `v0 + v1 + v2`.

- `v0`: first conservative synthetic set. Positives: AST normalization, docstring removal, local variable/argument renaming. Negatives: comparison flip, boolean-op flip, call-argument swap, wrong variable, small integer flip.
- `v1`: delta over `v0`. Positives: independent assignment reorder, boolean-return simplification, if-return to conditional expression, unreachable-else removal, and same-block import sorting. Negatives: membership/identity flip, condition negation, arithmetic-op flip, subscript-index flip, default-value flip, sort reverse flip, return-value removal, and await removal.
- `v2`: delta over `v1`. Positives: simple range-for to while-loop, list-append loop to comprehension, accumulator loop to `sum`, De Morgan boolean rewrite, and broader independent statement reorder. Negatives: loop-bound off-by-one, missing guard/edge-case branch, wrong exception type, dropped keyword argument, copy-vs-alias mutation, and dropped context manager/resource handling.

## Per-head training labels

| Pair type | Strategy head | Semantic head | Local head |
|---|---:|---:|---|
| formatting / rename / comments | close | close | low change |
| behavior-preserving refactor | close | close | aligned |
| `<` vs `<=`, off-by-one | close | far | changed span important |
| wrong variable / swapped args | close | far | changed span important |
| same task, different accepted solution | far/maybe | close | diffuse |
| unrelated code | far | far | high difference |

Possible loss structure:

```text
L = L_JEPA + lambda_pos L_pos + lambda_neg L_rank + lambda_local L_local + lambda_sigreg L_SIGReg
```

Ranking example:

```text
max(0, margin + sim(E(y), E(y_neg)) - sim(E(y), E(y_pos)))
```

The project should avoid the failure mode where a global embedding treats `i < n` and `i <= n` as merely almost identical.

## Downstream uses

### 1. Reference-known RLJF / reranking

Given prompt `x`, reference solution `y_ref`, and sampled candidates `y_1...y_k`:

```text
score_i = sim(z_semantic(y_i), z_semantic(y_ref))
```

Use top/bottom candidates as preference pairs for DPO-style post-training or as reranking output. Tests stay evaluation-only in the cleanest RLJF setup.

### 2. Hindsight agent training when a solution is eventually reached

In pure self-training, `y_ref` is not known upfront. If an agent eventually reaches a verified/accepted solution:

```text
y_1 fails, y_2 fails, ..., y_T succeeds
hindsight reference y_ref := y_T
```

Then earlier attempts can be ranked by similarity to the final solution and by whether they repeat known failure clusters. This is hindsight Code-JEPA feedback.

### 3. No-solution-yet failure memory

If all current candidates fail, Code-JEPA still helps, but only as exploration memory:

```text
M_bad = clusters of failed attempts in strategy/semantic/local space
```

For a new candidate:

```text
high strategy similarity + high semantic similarity to bad cluster -> duplicate failure / rephrase
high strategy similarity + low semantic similarity                 -> possible meaningful fix
low strategy similarity                                            -> new solution family
```

This rejects candidates that are only rephrasings of the same failed program. It does not certify that a novel candidate is correct. Novelty must be constrained by prompt fit, compile/static checks, type checks, or later verification.

### 4. Semantic duplicate rejection during self-training

When a generator samples many candidates for the same task, most may be rephrasings of the same algorithm or the same bug. Code-JEPA can cluster generated candidates before testing/judging:

```text
20 samples -> 4 semantic/strategy clusters -> keep representatives
```

Use cases:

- reject candidates that are just surface rewrites of known bad attempts;
- improve candidate-set diversity before expensive verification;
- estimate unique solution-family coverage rather than raw sample count.

Metrics:

- unique failure/solution clusters per token budget;
- pass@k per unique cluster;
- same solve rate with fewer tests/rollouts;
- duplicate-rejection precision: do not reject small real fixes as rephrases.

### 5. Out-of-the-box embedding tasks

Code search and clone detection should be treated as cheap downstream checks that come almost for free from the Siamese embedding setup:

```text
code search:     query/code -> embeddings -> nearest neighbors
clone detection: code/code  -> embeddings -> threshold or retrieval ranking
```

These tasks are not the core claim, but they are useful sanity checks and baseline comparisons because they require no agent loop, no verifier, and no decoder.

### 6. Cross-language retrieval and code translation

Use cross-language code-to-code retrieval as a bridge to code translation. A simple future task is Python-to-Java translation in latent space:

```text
Python context code -> Python context encoder -> z_py
Java target code    -> Java target encoder    -> z_java
z_py -> predictor -> predicted z_java
```

The predictor learns the Python-to-Java semantic mapping. This is a downstream/ablation setup, not a replacement for the default shared-encoder Code-JEPA training. Compare it against the simpler Siamese shared-encoder retrieval setup where Python and Java snippets are embedded directly and matched by cosine similarity.

Evaluation should start with CodeNet-style same-problem Python/Java pairs and measure retrieval MAP/MRR before adding any decoder. If generation is added later, condition a Java generator on retrieved neighbors or the predicted Java-space embedding and compare against standard translation models.

Ablations/baselines:

- shared Siamese Code-JEPA retrieval vs Python-context/Java-target predictor mapping;
- frozen vs finetuned encoders;
- `h` embedding vs projected `z` space;
- lexical and AST baselines;
- CodeBERT / GraphCodeBERT / UniXcoder / CodeT5-style embeddings;
- supervised seq2seq code translation where available.

## Evaluation and baselines

The first scientific bar is not DPO. It is proving that Code-JEPA is a better candidate judge/ranker than obvious baselines.

Required comparisons:

- base generator;
- supervised fine-tuning baseline;
- lexical/reference heuristics;
- CodeBERTScore / CodeBERT / UniXcoder-style embedding rankers;
- generic embedding rankers;
- Code-JEPA reranking;
- DPO with generic/code-embedding preferences;
- RLJF with frozen Code-JEPA preferences.

Useful evaluation axes:

- candidate reranking quality;
- pass@1 improvement after reranking;
- robustness to refactors/renames/formatting;
- discrimination of hard negatives;
- failure-cluster duplicate detection;
- whether small meaningful fixes are kept instead of rejected as rephrases.

Expected gains may be modest in pass@1. The stronger story is representation quality: semantic reranking, hard-negative sensitivity, and agent failure-memory usefulness.

Current empirical note: on v0 synthetic triples, base CodeBERT mean-pool is weak (`~0.30` rank_acc). The old pretrained CodeBERT JEPA finetune reached `~0.75` mean-pool / `~0.78` predictor on a same-sample comparison. A scratch 6-layer RoBERTa with custom 16k BPE, Siamese loss, and SIGReg reached `0.7566` after 2h and `0.9217` after a 6h continuation on train-distribution v0 triples. This proves the easy-transform signal is learnable from scratch, but it is not yet evidence of general semantic judging.

## Data sources

Start with full six-language CodeSearchNet for function-level multilingual pipeline validation: Python, Java, JavaScript, Go, PHP, and Ruby.

Practical order:

1. Full CodeSearchNet for function/method-level synthetic transform training and cross-language sanity checks.
2. CodeParrot clean Python as the public non-gated whole-file fallback when The Stack / StarCoderData auth is unavailable.
3. Larger permissive whole-file corpora from The Stack / StarCoderData / similar once license filtering and storage are decided.
4. Task/reference corpora for real semantic positives and cross-language retrieval: HumanEval, MBPP, APPS, CodeContests, CodeNet-like multi-solution datasets if accessible.

Split by repository/source/task, never by transformed view. A unit and all derived views must remain in the same split. Task datasets should emit accepted-solution semantic pairs when multiple correct implementations or multiple languages exist for the same problem.

## Risks and settled constraints

- No reference or verifier means no correctness signal. Code-JEPA can only avoid known-bad regions and encourage diversity.
- Hard negatives are behavior-impacting mutations, not always proven wrong for an unknown spec.
- False positives are dangerous: unsafe “equivalent” transforms can poison invariance.
- Pure novelty can drift into irrelevant code; keep sanity/prompt-fit gates.
- A single global vector is too blunt for one-character semantic changes; keep local span training.
- Do not spend CPU time on exact tokenization until the backbone/tokenizer is chosen.

## Related papers injected into this repo

Paper summaries live in `docs/paper_summaries/`:

- LeJEPA: `docs/paper_summaries/summary_2511.08544_lejepa/summary.md`
- LeWorldModel: `docs/paper_summaries/summary_2603.19312_leworldmodel/summary.md`
- LLM-JEPA: `docs/paper_summaries/summary_2509.14252_llm_jepa/summary.md`
- UniXcoder: `docs/paper_summaries/summary_2203.03850_unixcoder/summary.md`

The duplication/literature-risk note is in `docs/literature/exa-duplication-review.md`.
