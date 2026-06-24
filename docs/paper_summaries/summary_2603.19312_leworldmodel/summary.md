# LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

arXiv: https://arxiv.org/abs/2603.19312

Source fetched with `~/bin/arxiv-src 2603.19312`; entrypoint observed at `/tmp/arxiv-src/2603.19312/neurips_2026.tex`.

## Problem and core idea

LeWorldModel trains a JEPA world model directly from pixels. It learns an encoder from observations to latent states and a predictor that rolls latent states forward under actions. Training uses next-embedding prediction plus a Gaussian latent regularizer.

## Method details

- Encoder maps observations to compact latent representations.
- Predictor models latent dynamics: current latent plus action predicts next latent.
- Planning is performed in latent space by rolling candidate action sequences and selecting trajectories whose final latent state is closest to the goal.
- The video discussion around Push-T uses CEM-style latent trajectory search, not MCTS.

## Relevance for Code-JEPA

The analogy is not that code has physical dynamics. The useful idea is latent-space search/memory: a learned embedding space can support planning, candidate comparison, and surprise/failure detection. For code agents, Code-JEPA can serve as a memory of failed code regions and a similarity/ranking model over candidate programs.

## Concrete experiments to run

- Use Code-JEPA embeddings to cluster failed attempts and reject duplicate failure families.
- Test novelty-vs-correctness tradeoffs when no solution has been found.
- Compare global semantic similarity with local/span-sensitive similarity for small fixes.

## Risks / open questions

- World-model planning has a verifier-like goal state; code self-training often lacks a known target.
- Latent novelty alone can produce irrelevant code.
- Without tests, accepted patches, or hindsight success, Code-JEPA cannot prove correctness.
