# LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics

arXiv: https://arxiv.org/abs/2511.08544

Source fetched with `~/bin/arxiv-src 2511.08544`; entrypoint observed at `/tmp/arxiv-src/2511.08544/main.tex`.

## Problem and core idea

LeJEPA gives a cleaner JEPA objective by adding Sketched Isotropic Gaussian Regularization (SIGReg). The claim is that an isotropic Gaussian embedding distribution is theoretically preferred for downstream prediction risk, and SIGReg prevents collapse without common JEPA heuristics.

## Method details

- JEPA predictive loss in latent space.
- SIGReg regularizes embeddings toward an isotropic Gaussian using random projections / sketched normality checks.
- Claimed benefits: one main tradeoff hyperparameter, linear time/memory, stable across architectures and domains, no stop-grad/EMA/teacher-student requirement.

## Relevance for Code-JEPA

LeJEPA is the main candidate for the anti-collapse/regularization side of Code-JEPA. The code domain adds a harder requirement: invariance to harmless rewrites while separating tiny behavior-changing edits. SIGReg may solve collapse, but not hard-negative sensitivity by itself.

## Concrete experiments to run

- Compare EMA target-encoder JEPA vs SIGReg-style LeJEPA on the same code units.
- Measure embedding isotropy/collapse metrics during training.
- Test whether SIGReg improves downstream reranking or only stabilizes training.
- Check if hard-negative margin losses conflict with Gaussian regularization.

## Risks / open questions

- LeJEPA is vision/general SSL oriented, not code-specific.
- SIGReg does not directly teach `<` vs `<=` sensitivity.
- We need empirical proof that the regularizer helps once semantic/local hard negatives are added.
