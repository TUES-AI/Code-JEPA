# Exa duplication review: RLJF / Code-JEPA Feedback

## Verdict

No exact duplicate was found for:

> a frozen code-native JEPA model used as reference-conditioned feedback to create DPO/RLHF-style preferences for code generation, while keeping tests out of training.

But the surrounding space is crowded. The idea is plausible for a small workshop/arXiv paper if positioned carefully and benchmarked against strong embedding, reranking, and code-DPO baselines.

The safest non-overclaiming position:

> RLJF studies whether a code-native JEPA embedding space can serve as an execution-free, reference-conditioned preference signal for generated code, improving over token-level SFT and generic embedding feedback while remaining complementary to test-based RL.

## Closest work

### 1. CodeBERTScore

Link: https://aclanthology.org/2023.emnlp-main.859/

Very important baseline. CodeBERTScore evaluates generated code by comparing it to reference code using pretrained code-model embeddings, with NL context. It correlates better with human preference and functional correctness than token metrics.

Closeness:

- Similar: reference-conditioned semantic similarity for code.
- Different: evaluation metric, not Code-JEPA-trained feedback for DPO/post-training.
- Reviewer demand: compare RLJF against CodeBERTScore-based reranking and CodeBERTScore-based preference generation.

### 2. RefAlign / PGSRM / semantic-similarity rewards

Representative links:

- PGSRM: https://arxiv.org/html/2512.06920
- RefAlign: https://openreview.net/forum?id=BaFPkXHdrg

These use reference-output embedding similarity as a reward/surrogate signal for LLM alignment.

Closeness:

- Similar: generated output is rewarded by embedding similarity to a reference answer.
- Different: not code-specific, not JEPA, not using code structure/hard negatives.
- Risk: RLJF can look like “CodeBERTScore/embedding reward + DPO.” Need emphasize code-native JEPA training and code-specific invariances.

### 3. CodeDPO and code preference optimization

Links:

- CodeDPO: https://aclanthology.org/2025.acl-long.771/
- Aligning CodeLLMs with DPO: https://arxiv.org/abs/2410.18585
- Code-Optimise: https://arxiv.org/abs/2406.12502
- DSTC: https://arxiv.org/abs/2411.13611

These train code models with preference optimization, mostly using generated tests, execution, correctness, efficiency, or synthetic validation.

Closeness:

- Similar: code LLM post-training with DPO/preferences.
- Different: preferences are usually execution/test/self-validation based, not frozen JEPA similarity to reference.
- Reviewer demand: compare against at least one test/self-validation preference method if compute allows, or clearly state RLJF is execution-free and complementary.

### 4. Code preference model work

Links:

- CodeFavor / CodePrefBench: https://llm-code-preference.github.io/
- Repo: https://github.com/amazon-science/llm-code-preference

This trains and evaluates code preference models from synthetic evolution, covering correctness, efficiency, security, and human preferences.

Closeness:

- Similar: learned judge for code quality/preferences.
- Different: pairwise preference model, not JEPA embedding similarity judge.
- Reviewer demand: compare Code-JEPA judge against a code preference model on pairwise ranking if feasible.

### 5. Reranking generated code

Links:

- Coder-Reviewer: https://arxiv.org/abs/2211.16490
- SRank / Functional Overlap Reranking: https://aclanthology.org/2024.findings-acl.220/
- SolveRank: https://aclanthology.org/2025.findings-emnlp.281/
- CodeRSA: https://arxiv.org/abs/2502.15835

These rerank multiple generated code candidates using reviewer likelihood, functional overlap, solution-aware retrieval/ranking, or pragmatic reasoning.

Closeness:

- Similar: select best candidate among generated samples.
- Different: RLJF uses frozen Code-JEPA similarity to a reference and then uses the same signal for DPO-style post-training.
- Reviewer demand: include reranking-only baselines before claiming post-training benefit.

### 6. LLM-JEPA / JEPA for language/code

Links:

- LLM-JEPA: https://arxiv.org/abs/2509.14252
- Repo: https://github.com/galilai-group/llm-jepa
- Small JEPA-CODE repo: https://github.com/Mahdi-Rashidiyan/JEPA-CODE

LLM-JEPA adds JEPA-style embedding objectives to LLM pretraining/fine-tuning with paired views such as text/code.

Closeness:

- Similar: JEPA adapted to LLM/code-like tasks.
- Different: RLJF is not an auxiliary JEPA loss in the generator; it uses a frozen Code-JEPA as feedback for candidate preferences.

### 7. RLCF: Coarse-Tuning Models of Code with Reinforcement Learning Feedback

Link: https://arxiv.org/abs/2305.18341

Uses compiler feedback and a separate LLM comparing generated code to reference code.

Closeness:

- Similar: RL for code with reference-comparison feedback.
- Different: not JEPA, mixes compiler/test-like feedback, and uses an LLM comparator rather than a code-native embedding judge.

### 8. Execution-free/static-analysis reward work

Link: https://arxiv.org/html/2605.17174

Studies execution-free rewards for code RL, including static checking and similarity-based rewards.

Closeness:

- Similar: non-execution reward design for code generation.
- Different: diffusion-code RL setting; not Code-JEPA reference-conditioned DPO.
- Important: if this paper reports similarity rewards only help on easier tasks, RLJF should examine task difficulty splits.

## Main duplication risk

The biggest risk is not direct JEPA duplication. It is reviewers saying:

> This is just CodeBERTScore or generic embedding similarity used to make DPO preferences.

To avoid that, the paper needs to prove at least one of:

1. Code-JEPA ranks generated code candidates better than CodeBERTScore/generic code embeddings.
2. Code-JEPA is more invariant to semantics-preserving transformations while sensitive to hard semantic bugs.
3. RLJF improves DPO/reranking specifically because of Code-JEPA training, not just because any embedding similarity to reference works.

## Required ablations/baselines

Minimum:

1. Base model.
2. SFT on reference solutions.
3. DPO with lexical or exact-reference preference heuristic.
4. DPO with CodeBERTScore preferences.
5. DPO with generic code embedding similarity, e.g. CodeBERT/GraphCodeBERT/UniXcoder/jina-code-embeddings.
6. RLJF with Code-JEPA.
7. Reranking-only variants for the same scoring methods.

Strong additions:

- CodeFavor/code-preference-model judge baseline.
- Coder-Reviewer or SRank reranking baseline.
- A small test-based DPO/CodeDPO-style baseline under matched sample budget.
- Different Code-JEPA training recipes: augmentations only vs AST/DFG vs hard negatives.
- Difficulty split: easy vs hard tasks.
- Reference-style perturbation: candidates/reference use different style but same behavior.

## Expected results

Likely not shocking.

Realistic expectation:

- reranking gains are easier and probably larger;
- DPO/RLJF gains likely modest on HumanEval/MBPP, maybe 1–3 pass@1 points if implementation is solid;
- stronger gains may appear on robustness metrics: refactor invariance, style mismatch, hard-negative ranking, or candidate preference accuracy;
- Code-JEPA may help most when tests are absent or references differ stylistically from candidates.

A strong paper should not rely only on pass@1. It should show that Code-JEPA provides a different kind of signal.

## What RLJF can offer that normal methods cannot

### Execution-free training feedback

Unlike test-based RL, RLJF can operate when no tests/sandbox are available, as long as reference or high-quality code exists.

### Reference-conditioned preferences from one solution

Given one reference, RLJF can rank many generated candidates without human pairwise labels.

### Invariance to superficial code changes

If Code-JEPA is trained well, it can reward candidates that differ lexically from the reference but preserve semantic/structural intent.

### Sensitivity to code-native hard negatives

Unlike generic text embeddings, Code-JEPA can be trained to separate near-miss bugs that look textually similar:

```text
off-by-one
wrong operator
wrong variable
wrong API order
missing edge case
same AST shape, wrong behavior
```

### Same judge for reranking and post-training

The frozen Code-JEPA can support:

```text
candidate reranking
DPO preference generation
retrieval of similar solutions
embedding-space diagnostics
quality/error analysis
```

### Clean separation from verifiable RL

Tests are only evaluation, not training. This isolates whether learned code embeddings can provide useful feedback by themselves.

## Best small-publication framing

Do not claim:

> We beat verifiable RL for code.

Claim:

> We introduce and evaluate Code-JEPA feedback as an execution-free, reference-conditioned preference signal for code generation. Compared with token-level imitation and generic embedding feedback, Code-JEPA improves candidate ranking and modestly improves DPO-style post-training, while remaining complementary to test-based RL.
