# LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures

arXiv: https://arxiv.org/abs/2509.14252

Source fetched with `~/bin/arxiv-src 2509.14252`; entrypoint observed at `/tmp/arxiv-src/2509.14252/main.tex`.

## Problem and core idea

LLM-JEPA applies JEPA-style embedding-space training to language models, arguing that LLM training is over-focused on input-space reconstruction while JEPA objectives can improve representation learning and robustness.

## Method details

- Embedding-space predictive objective for LLM pretraining/fine-tuning.
- Reported improvements across datasets including NL-RX, GSM8K, Spider, RottenTomatoes and model families including Llama3, OpenELM, Gemma2, Olmo.
- Important precedent: JEPA-style objectives can be adapted beyond vision.

## Relevance for Code-JEPA

LLM-JEPA is the closest general-language neighbor, but Code-JEPA differs by using code-specific transformations and hard negatives. Our key gap is not merely “JEPA for text/code”; it is hard-negative-sensitive code geometry used as frozen feedback/memory for candidate generation and agent self-training.

## Concrete experiments to run

- Reproduce a small embedding-space JEPA objective on code tokens/functions.
- Compare against normal masked/next-token or encoder embedding baselines.
- Add code-specific positive and hard-negative transformations and measure whether they change ranking quality.

## Risks / open questions

- If Code-JEPA only behaves like a generic language embedding, reviewers will collapse it into CodeBERTScore/generic embedding feedback.
- Need baselines against CodeBERT, UniXcoder, CodeBERTScore, and generic embeddings before RLJF claims are compelling.
