# RLJF: Reinforcement Learning from JEPA Feedback for Code Generation

## One-line idea

Train a code-native JEPA embedding model, freeze it, and use it as an automatic semantic/structural judge to create preference pairs for code LLM post-training.

Short name: **RLJF** — Reinforcement Learning from JEPA Feedback.

## Motivation

Supervised fine-tuning for code generation usually trains a model to match one reference implementation token-by-token. This is overly rigid because many different implementations can be correct, idiomatic, and structurally reasonable.

Test-based RL or verifiable RL solves part of this by using execution feedback, but it mixes multiple concerns:

- correctness from tests;
- reward design;
- rollout/search budget;
- code style/structure;
- semantic similarity to known good solutions;
- policy optimization details.

RLJF intentionally separates concerns. The training feedback comes from a frozen Code-JEPA model, not from test execution. Tests are reserved for evaluation.

The core question is:

> Can a code-native JEPA embedding space provide a useful preference signal for code LLM post-training, beyond exact-token imitation?

## Code-JEPA component

The Code-JEPA model maps code to a latent embedding:

```text
E(code) -> z_code
```

It should be trained before RLJF, then frozen.

The target is not just lexical similarity. The embedding space should be invariant to superficial differences while preserving important program properties.

Possible self-supervised training views:

```text
raw code <-> formatted code
raw code <-> variable-renamed code
raw code <-> normalized code
raw code <-> AST serialization
raw code <-> data-flow representation
raw code <-> control-flow representation
function <-> call-site context
old code <-> refactored equivalent code
implementation A <-> behaviorally equivalent implementation B
```

## Code permutation and augmentation bank

For Code-JEPA training, use two kinds of transformations:

```text
positive augmentations = same behavior, different surface/structure
hard negatives         = tiny change, different behavior
```

Positive augmentations teach invariance. Hard negatives teach sensitivity.

### Positive augmentations

#### Surface-level rewrites

Easy and safe first-pass augmentations:

- variable renaming;
- internal function/class renaming when references can be updated safely;
- formatting with Black/YAPF/autopep8/Prettier;
- quote style changes;
- whitespace changes;
- comment removal or rewrite;
- docstring paraphrase/removal;
- numeric literal style changes, e.g. `1000` ↔ `1_000`, `0xff` ↔ `255`.

#### Import/dependency normalization

- reorder imports;
- split/merge imports:

```python
import os, sys
```

↔

```python
import os
import sys
```

- `import module` ↔ `from module import name` when safe;
- alias normalization, e.g. canonical handling of `import numpy as np`.

#### Function/class reordering

Top-level definitions can often be reordered if there are no order-dependent side effects:

```python
def a(): ...
def b(): ...
class C: ...
```

↔

```python
class C: ...
def b(): ...
def a(): ...
```

For Python, be conservative around decorators, metaclasses, registrations, monkey-patching, and top-level executable code.

#### Independent statement reordering

If dependency/effect analysis says two statements do not depend on each other, they can be swapped:

```python
a = f(x)
b = g(y)
c = a + b
```

`a = ...` and `b = ...` can be swapped if `f` and `g` are pure or whitelisted safe.

Conservative requirements:

- no shared variable writes;
- no mutation of the same object;
- no I/O;
- no random/time/global state;
- no function calls unless whitelisted pure.

#### Equivalent syntax rewrites

```python
if cond:
    return True
else:
    return False
```

↔

```python
return bool(cond)
```

```python
x = []
for a in arr:
    x.append(f(a))
```

↔

```python
x = [f(a) for a in arr]
```

Other candidates:

- `for i in range(len(xs))` ↔ `for i, x in enumerate(xs)`;
- `dict()` ↔ `{}`;
- `list()` ↔ `[]`;
- simple guard clause ↔ nested conditional;
- early return ↔ wrapped block;
- `if x is None` normalization.

#### Algebraic/logical rewrites

When types and side effects are safe:

```text
a + b ↔ b + a
x * 1 ↔ x
x + 0 ↔ x
not (a or b) ↔ (not a and not b)
```

Be careful with Python operator overloads and short-circuit effects. These are safest in restricted/math-heavy code.

#### Control-flow preserving rewrites

- invert condition and swap branches:

```python
if cond:
    A
else:
    B
```

↔

```python
if not cond:
    B
else:
    A
```

- while loop ↔ for loop in simple cases;
- early `continue` ↔ nested `if`;
- temporary variable introduction/removal:

```python
return f(x)
```

↔

```python
y = f(x)
return y
```

#### Function extraction / inlining

- extract a block into a helper function;
- inline tiny helpers;
- split large function into helper calls;
- merge trivial helper into caller.

This is powerful but should be conservative around closures, mutation, exceptions, decorators, and public APIs.

#### Type annotation changes

- add/remove type hints;
- normalize `List[int]` ↔ `list[int]`;
- add simple inferred annotations;
- remove redundant annotations.

#### Repo/module structure views

These are not always mutations, but they are strong alternate views for Code-JEPA:

- file ↔ dependency graph;
- function ↔ call graph neighborhood;
- class ↔ method graph;
- code ↔ AST serialization;
- code ↔ data-flow graph;
- code ↔ control-flow graph;
- file ↔ imported symbols;
- patch ↔ changed dependency graph;
- function ↔ call-site context.

The dependency-graph/reordering direction is especially promising because generic text embeddings are unlikely to exploit it well.

### Hard negatives

Hard negatives should look close but change behavior:

```text
off-by-one variant
wrong comparison operator
wrong variable used
wrong API order
missing null/edge-case handling
same AST shape but wrong behavior
compiles but fails hidden cases
```

Concrete hard-negative edits:

- `<` ↔ `<=`;
- `+1` ↔ `-1`;
- `and` ↔ `or`;
- wrong variable substitution;
- swapped argument order;
- remove edge-case branch;
- wrong default value;
- wrong loop boundary;
- mutate copy vs original;
- sort ascending vs descending;
- missing `return`;
- wrong exception handling;
- async call not awaited;
- API calls in wrong order.

These teach:

> structurally similar does not always mean semantically equivalent.

### MVP subset

For the first implementation, start with:

```text
1. format/comment/rename augmentations
2. import/function/class reordering
3. AST/DFG/CFG alternate views
4. hard negatives: operators, bounds, variable swaps
5. hidden tests only for evaluation
```

The exact collapse-prevention recipe is open. Candidate families:

- LeJEPA/SIGReg-style Gaussian regularization;
- Barlow Twins / VICReg-style decorrelation regularization;
- contrastive negatives;
- code-specific hard-negative objectives.

## RLJF training setup

Given a prompt and a reference solution:

```text
prompt x: "write code that does XYZ"
reference code y_ref
```

Sample multiple candidate outputs from the current policy:

```text
y_1, y_2, ..., y_k ~ πθ(. | x)
```

Score each candidate with frozen Code-JEPA similarity to the reference:

```text
score_i = sim(E(y_i), E(y_ref))
```

Then create preference pairs:

```text
preferred = candidate with high Code-JEPA score
rejected  = candidate with low Code-JEPA score
```

Use a preference optimization method such as DPO-style training:

```text
πθ should prefer y_preferred over y_rejected for prompt x
```

This is the cleanest version. PPO/GRPO-style variants are possible but not necessary for the first paper.

## Important design choice

Do **not** use tests during RLJF training if the goal is to isolate the Code-JEPA contribution.

Use tests only for evaluation.

This gives a clean separation:

```text
Training signal: Code-JEPA feedback only
Evaluation signal: compile rate, pass@k, benchmark tests, code quality metrics
```

## Baselines

Minimum baselines:

1. Base code LLM without post-training.
2. SFT on reference code.
3. DPO with lexical/reference-based preference heuristic.
4. Reranking with CodeBERTScore.
5. Reranking with generic code embeddings, e.g. CodeBERT/GraphCodeBERT/UniXcoder/jina-code-embeddings.
6. Reranking with Code-JEPA.
7. DPO using CodeBERTScore or generic embedding preferences.
8. RLJF: DPO using frozen Code-JEPA preference.

Optional stronger baselines:

- execution/test-based reranking;
- verifier model reranking;
- DPO using LLM-as-judge labels;
- code preference model judge, e.g. CodeFavor-style pairwise preference model;
- Coder-Reviewer / SRank-style reranking if feasible;
- GRPO/verifiable RL under matched sample budget.

## Evaluation

Use tests for evaluation, not training.

Metrics:

```text
pass@1 / pass@k
compile rate
syntax error rate
unit test success
exact match, but only as diagnostic
CodeBLEU / structural metrics
complexity / lint score
sample efficiency
preference accuracy against held-out human/test-derived labels
```

Candidate datasets:

```text
HumanEval
MBPP
APPS subset
CodeContests subset
Repo-level patch datasets if available
```

A practical first experiment:

```text
small code LLM + HumanEval/MBPP
sample 4-16 completions per prompt
RLJF-DPO from Code-JEPA preferences
evaluate pass@1 and pass@k against SFT and reranking baselines
```

## Expected results

The expected gains are probably modest, not shocking.

A realistic outcome:

- small pass@1 improvement over SFT or base DPO;
- larger improvement in reranking quality;
- better tolerance of valid alternative implementations;
- fewer syntax/style/structure pathologies if Code-JEPA was trained with structural views;
- stronger benefits when references are stylistically different from generated solutions.

A likely paper claim is not “RLJF beats verifiable RL.”

A cleaner claim is:

> RLJF provides a non-execution semantic/structural feedback channel that improves code generation over exact-token imitation and generic embedding feedback, while remaining independent of unit tests during training.

## Key validation bar: smoke the encoder-style rankers

The first mission is not DPO. The first mission is to prove Code-JEPA is a better judge/ranker of generated code candidates than existing encoder-style metrics.

If RLJF does not beat or meaningfully differ from CodeBERTScore, CodeBERT, UniXcoder, or similar embedding rankers, reviewers can reasonably ask:

> Why not just use an existing code embedding metric?

Minimum smoke test:

```text
prompt + reference code
LLM generates K candidates
rank candidates by:
  1. CodeBERTScore
  2. CodeBERT/GraphCodeBERT/UniXcoder/jina-code-embedding cosine
  3. Code-JEPA cosine
  4. model logprob
  5. random
use hidden tests only for evaluation of the top-ranked candidate
```

The clearest win is not just higher pass@1. It is showing where Code-JEPA wins:

- reference and candidate are stylistically different;
- variables/functions are renamed;
- implementations are semantically same but lexically different;
- candidate is shorter/longer than reference;
- same AST shape but subtle bug;
- off-by-one, wrong operator, wrong variable, wrong API order;
- generated code has plausible structure but wrong behavior.

The strongest diagnostic claim:

> Code-JEPA is more invariant to harmless code transformations and more sensitive to behavior-changing edits than generic code embedding rankers.

Only after this ranker result is established does RLJF become the natural downstream use.

## What this may unlock beyond normal methods

### 1. Feedback when tests are unavailable

Many real code tasks do not have complete tests. RLJF can still provide a learned feedback signal from reference code, repo context, or high-quality examples.

### 2. Credit for valid alternative implementations

Exact SFT punishes deviations from the reference. RLJF can reward code that is different textually but close in learned semantic/structural space.

### 3. Structure/style feedback without human labels

If Code-JEPA is trained on AST/dataflow/refactoring views, it can reward structural organization that token likelihood may not capture cleanly.

### 4. Separation from execution-based RL

RLJF can be studied independently from test-based RL. This makes it useful as an ablation component and potentially complementary to verifiable RL later.

### 5. Reference-conditioned preference without hand-written reward models

Given a reference solution, RLJF automatically creates candidate preferences without requiring human preference labels.

### 6. Reranking and post-training from the same judge

The same frozen Code-JEPA model can be used for:

```text
candidate reranking
DPO preference generation
quality diagnostics
embedding-space analysis
retrieval of similar solutions
```

This makes it operationally simple.

## Main risks

### Risk 1: Code-JEPA similarity may not equal correctness

A candidate can be close to the reference embedding but still fail edge cases.

Mitigation:

- keep tests for evaluation;
- use hard negatives during Code-JEPA training;
- compare against generic code embedding models;
- report where RLJF improves vs where it fails.

### Risk 2: It may collapse into reference mimicry

If the Code-JEPA space is too lexical, RLJF becomes weak DPO toward the reference.

Mitigation:

- train with aggressive behavior-preserving transformations;
- use AST/dataflow/control-flow views;
- evaluate on refactored/equivalent-solution robustness.

### Risk 3: Existing embedding models may already do enough

If CodeBERTScore, CodeBERT, GraphCodeBERT, UniXcoder, or jina-code-embeddings work similarly, novelty is weaker.

Mitigation:

- include CodeBERTScore and generic embedding baselines;
- show Code-JEPA is better for preference ranking of generated code;
- show specific robustness wins on refactors, style mismatch, hard negatives, and behavior-changing near-misses;
- emphasize code-native augmentations and hard negatives.

### Risk 4: Gains may be small

Likely true. The paper should focus on clean separation of feedback sources, not on huge benchmark wins.

## Minimal paper shape

Title candidates:

- **RLJF: Reinforcement Learning from JEPA Feedback for Code Generation**
- **Code-JEPA Feedback for Preference Optimization in Code LLMs**
- **Semantic Preference Optimization for Code Generation with Code-JEPA**
- **Learning from Latent Code Similarity: JEPA Feedback for Code LLMs**

Core contributions:

1. Train a code-native JEPA embedding model using program transformations and structural views.
2. Show Code-JEPA beats encoder-style rankers such as CodeBERTScore and generic code embeddings on candidate ranking, especially under refactors, style mismatch, and hard negatives.
3. Use the frozen Code-JEPA as a feedback model to create preferences among generated code candidates.
4. Post-train a code LLM with DPO-style RLJF, using no tests during training.
5. Evaluate whether Code-JEPA feedback improves code generation and reranking compared with SFT, lexical preferences, and generic code embedding feedback.

## The cleanest claim

> Code-JEPA learns a code-native latent space where semantically and structurally related programs are close. RLJF uses this frozen latent space as automatic feedback for preference optimization, giving code LLMs a non-execution reward signal that is less brittle than token-level reference matching.
