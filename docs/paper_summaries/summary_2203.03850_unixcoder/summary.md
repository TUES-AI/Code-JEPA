# UniXcoder: Unified Cross-Modal Pre-training for Code Representation

arXiv: https://arxiv.org/abs/2203.03850

Source fetched with `~/bin/arxiv-src https://arxiv.org/pdf/2203.03850`; entrypoint observed at `/tmp/arxiv-src/2203.03850/acl_latex.tex`.

## Problem and core idea

UniXcoder is a unified Transformer for code understanding, generation, and completion. It uses attention masks plus mode prefixes to run the same parameters as encoder-only, decoder-only, or encoder-decoder. For code representations, it adds AST/comment signals and trains code-fragment embeddings with contrastive learning and comment generation.

## Method details

- Input can include code comments and a flattened AST sequence.
- The AST flattening is a one-to-one recursive mapping with left/right node markers, avoiding ambiguity from plain BFS/DFS traversal.
- Pre-training combines MLM, unidirectional LM, denoising, multi-modal contrastive learning, and cross-modal comment generation.
- Downstream embeddings are mean-pooled hidden states; clone detection, code search, and zero-shot code-to-code search use cosine similarity.
- AST is used during pre-training, then non-terminal AST symbols can be dropped for cheaper fine-tuning/inference.

## Key results

- UniXcoder reports best results among compared pretrained code models on clone detection and code search: POJ-104 MAP@R 90.52, BigCloneBench F1 95.2, CosQA MRR 70.1, AdvTest MRR 41.3, CSN MRR 74.4.
- On zero-shot code-to-code search over CodeNet Ruby/Python/Java, overall MAP is 20.45 versus GraphCodeBERT 9.17, CodeBERT 4.94, PLBART 8.25, and CodeT5-base 7.81.
- Ablations show representation contrastive learning and cross-modal generation matter strongly for code-to-code search; removing contrastive learning drops overall MAP from 20.45 to 13.73.

## What is relevant for Code-JEPA

UniXcoder is a strong baseline for Code-JEPA embedding evaluations, especially code search, clone detection, and cross-language code-to-code retrieval. Its zero-shot code-to-code search framing is directly relevant to future Python-to-Java translation experiments: retrieval can be the simplest translation-like downstream probe before adding a generator.

Code-JEPA differs by explicitly training behavior-preserving positives and behavior-impacting hard negatives. The expected advantage should be tested on tiny semantic edits and failure-memory tasks, not only on broad same-problem retrieval.

## Concrete experiments to run next

- Add UniXcoder to embedding baselines for code search, clone detection, and CodeNet-style code-to-code retrieval.
- Use CodeNet Python/Java same-problem pairs to train a translation-style JEPA predictor: Python context embedding -> predicted Java target embedding.
- Compare shared Siamese Code-JEPA retrieval against the Python-context / Java-target predictor setup.
- Evaluate whether Code-JEPA separates hard negatives better than UniXcoder while staying competitive on same-task cross-language retrieval.
- Report retrieval MAP/MRR before any decoder-based translation, then optionally condition a generator on the retrieved/predicted Java-neighbor embedding.

## Risks / open questions

- UniXcoder's reported code-to-code search MAP is still low in absolute terms, so cross-language embeddings remain hard.
- Same CodeNet problem labels do not guarantee exact algorithmic equivalence or idiomatic translation pairs.
- Comment-generation alignment may help broad semantics but may miss one-token behavior changes; this is where Code-JEPA hard negatives should matter.
