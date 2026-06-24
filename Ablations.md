# Ablations

Future experiment parking lot for once a usable Code-JEPA checkpoint exists. Keep this file about datasets, metrics, and ablation axes; keep `Project.md` about the scientific idea.

## Downstream tasks to evaluate

### 1. Failure-memory for code agents

Goal: avoid repeated failed solution families when no correct solution is known yet.

Datasets/options:

- HumanEval/MBPP with generated failed attempts;
- APPS/CodeContests if we need harder multi-step attempts;
- saved agent trajectories from our own coding agents later.

Metrics:

- solve rate under fixed rollout budget;
- tests/verifier calls saved;
- duplicate-failure rejection precision/recall;
- number of unique semantic/strategy clusters explored;
- false rejection rate for small real fixes.

### 2. Hindsight learning from long agent trajectories

Goal: when an agent eventually reaches a verified solution, use the final solution as a hindsight anchor for earlier attempts.

Datasets/options:

- synthetic repair trajectories generated from benchmark tasks;
- real agent logs once available;
- accepted solutions plus mutated/intermediate variants.

Metrics:

- steps-to-success reduction;
- preference accuracy over trajectory states;
- ranking correlation with final verified solution;
- improvement after DPO/RLJF on hindsight pairs.

### 3. Semantic duplicate rejection during self-training

Goal: when sampling many candidate solutions, keep semantically distinct candidates and reject rephrasings of known-bad attempts.

Datasets/options:

- generated samples from Qwen/CodeLlama-style coder models on HumanEval/MBPP/APPS;
- clusters of failed attempts from self-training runs.

Metrics:

- unique solution-family count per generated token;
- pass@k per unique cluster;
- same pass rate with fewer candidates tested;
- cluster purity for known hard-negative transformations.

## Core representation ablations

| Ablation | Question |
|---|---|
| JEPA only | Does predictive latent training alone learn useful code geometry? |
| JEPA + positives | Do behavior-preserving transforms improve invariance? |
| JEPA + hard negatives | Do mutation negatives improve one-character semantic sensitivity? |
| JEPA + positives + hard negatives | Main expected model. |
| with/without local head | Is local span supervision necessary for `<` vs `<=` style changes? |
| with/without strategy head | Does separate strategy space help failure clustering? |
| EMA target vs SIGReg | Which anti-collapse method is better for code? |
| in-batch contrastive/rank loss on/off | Does it prevent collapse or distort JEPA geometry? |

## Transformation ablations

Start conservative and add riskier transforms separately.

| Transform group | Role | Measure |
|---|---|---|
| rename/format/docstring | positive | refactor invariance |
| AST normalization | positive | syntax/format robustness |
| comparator/bool/int mutations | negative | hard-negative discrimination |
| wrong variable / swapped args | negative | semantic sensitivity |
| import/function reorder | positive risky | false-positive poisoning |
| syntax-equivalent rewrites | positive risky | invariance vs safety |
| DFG/CFG/call graph views | alternate view | downstream ranking gain |

## Baselines

- lexical/edit-distance similarity;
- CodeBERTScore;
- CodeBERT / UniXcoder embeddings;
- generic text/code embeddings;
- base generator without reranking;
- DPO/RLJF using generic embedding preferences.

## Notes

- Tests/verifiers are evaluation anchors, not necessarily training signals.
- Hard negatives are behavior-impacting mutations relative to original code, not guaranteed benchmark failures.
- Report storage/throughput for each data-prep run so data scale claims are auditable.
