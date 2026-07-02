#!/usr/bin/env python3
"""JAX fine-tuning of pretrained Siamese Code-JEPA on clone benchmarks.

Loads a `code-jepa-jax-multigpu-train-state-v1` checkpoint (the encoder + projection
head trained by `siamese_bpe_jepa_multigpu.py`) and fine-tunes it on:

- POJ-104 (clone retrieval): triplet margin loss, evaluated with MAP@R.
- BigCloneBench (clone detection): a small pair classifier on top of the encoder,
  evaluated with best-F1 / ROC-AUC / average precision.

The encoder embedding used downstream is selectable with `--embedding`:
- `z` (default): full encoder + projection head output (the space the model was
  trained to make discriminative).
- `h`: pre-projection mask-aware mean pool (Project.md's frozen search-embedding
  candidate).
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import optax
    from flax import linen as nn
    from flax import serialization
    from flax.traverse_util import path_aware_map
    from flax.training import train_state
except Exception as exc:  # pragma: no cover - exercised on GPU boxes.
    raise SystemExit(
        "JAX fine-tuning requires jax, flax, and optax. Use the Code-JEPA GPU env. "
        f"Import error: {type(exc).__name__}: {exc}"
    ) from exc

from transformers import PreTrainedTokenizerFast

from code_jepa.training.siamese_bpe_jepa import SiameseEncoder, precision_dtype


CHECKPOINT_FORMAT = "code-jepa-jax-multigpu-train-state-v1"


@dataclass(frozen=True)
class FinetuneConfig:
    benchmark: str
    benchmark_dir: str
    checkpoint: str
    output_dir: str
    tokenizer_path: str
    embedding: str = "z"  # z | h
    max_len: int = 256
    batch_size: int = 32
    eval_batch_size: int = 128
    epochs: int = 2
    lr: float = 2e-5
    head_lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    end_lr_ratio: float = 0.1
    margin: float = 0.2
    dropout: float = 0.1
    precision: str = "bf16"
    seed: int = 123456
    max_train_examples: int = 0
    max_valid_examples: int = 0
    max_test_examples: int = 0
    grad_clip: float = 1.0
    freeze_encoder: bool = False
    log_every: int = 50


@dataclass(frozen=True)
class EncoderCfg:
    """Duck-typed config exposing exactly the fields SiameseEncoder reads."""

    precision: str
    pad_token_id: int
    vocab_size: int
    hidden_size: int
    max_len: int
    layers: int
    heads: int
    intermediate_size: int
    dropout: float
    projection_dim: int


class EmbeddingModel(nn.Module):
    cfg: EncoderCfg
    use_hidden: bool

    @nn.compact
    def __call__(self, tokens: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
        out = SiameseEncoder(self.cfg)(tokens, deterministic=deterministic, return_hidden=self.use_hidden)
        return out[1] if self.use_hidden else out


class CloneClassifier(nn.Module):
    cfg: EncoderCfg
    use_hidden: bool
    head_dim: int
    dropout: float

    @nn.compact
    def __call__(
        self, left: jnp.ndarray, right: jnp.ndarray, *, deterministic: bool
    ) -> jnp.ndarray:
        encoder = SiameseEncoder(self.cfg)

        def embed(tokens: jnp.ndarray) -> jnp.ndarray:
            out = encoder(tokens, deterministic=deterministic, return_hidden=self.use_hidden)
            return out[1] if self.use_hidden else out

        left_vec = embed(left)
        right_vec = embed(right)
        features = jnp.concatenate(
            [left_vec, right_vec, jnp.abs(left_vec - right_vec), left_vec * right_vec], axis=-1
        )
        x = nn.Dropout(self.dropout)(features, deterministic=deterministic)
        x = nn.Dense(self.head_dim, name="head_dense0")(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout)(x, deterministic=deterministic)
        x = nn.Dense(1, name="head_dense1")(x)
        return x[..., 0]


class TrainState(train_state.TrainState):
    pass


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_pretrained_encoder_params(checkpoint_path: Path) -> tuple[dict[str, Any], EncoderCfg]:
    with checkpoint_path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"unexpected checkpoint format {payload.get('format')!r}; expected {CHECKPOINT_FORMAT!r}"
        )
    restored = serialization.msgpack_restore(payload["state"])
    params = restored["params"]
    if "SiameseEncoder_0" not in params:
        raise ValueError(f"checkpoint params missing SiameseEncoder_0; got {list(params)}")
    cfg = payload["config"]
    encoder_cfg = EncoderCfg(
        precision=str(cfg["precision"]),
        pad_token_id=int(cfg["pad_token_id"]),
        vocab_size=int(cfg["vocab_size"]),
        hidden_size=int(cfg["hidden_size"]),
        max_len=int(cfg["max_len"]),
        layers=int(cfg["layers"]),
        heads=int(cfg["heads"]),
        intermediate_size=int(cfg["intermediate_size"]),
        dropout=0.0,
        projection_dim=int(cfg["projection_dim"]),
    )
    encoder_params = jax.tree_util.tree_map(jnp.asarray, params["SiameseEncoder_0"])
    return encoder_params, encoder_cfg


def graft_encoder(fresh_params: Any, pretrained_encoder: dict[str, Any]) -> Any:
    params = dict(fresh_params)
    params["SiameseEncoder_0"] = pretrained_encoder
    return params


# ---------------------------------------------------------------------------
# Optimizer (per-group learning rates, optional frozen encoder)
# ---------------------------------------------------------------------------

def build_optimizer(params: Any, cfg: FinetuneConfig, total_steps: int) -> optax.GradientTransformation:
    warmup = max(1, int(total_steps * cfg.warmup_ratio))
    decay = max(warmup + 1, total_steps)

    def schedule(peak: float) -> optax.Schedule:
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=peak,
            warmup_steps=warmup,
            decay_steps=decay,
            end_value=peak * cfg.end_lr_ratio,
        )

    head_tx = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(learning_rate=schedule(cfg.head_lr), weight_decay=cfg.weight_decay),
    )
    if cfg.freeze_encoder:
        encoder_tx = optax.set_to_zero()
    else:
        encoder_tx = optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip),
            optax.adamw(learning_rate=schedule(cfg.lr), weight_decay=cfg.weight_decay),
        )
    labels = path_aware_map(
        lambda path, _: "encoder" if "SiameseEncoder_0" in path else "head", params
    )
    return optax.multi_transform({"encoder": encoder_tx, "head": head_tx}, labels)


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_fixed(
    tokenizer: PreTrainedTokenizerFast, texts: list[str], max_len: int
) -> np.ndarray:
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_attention_mask=False,
    )["input_ids"]
    return np.asarray(encoded, dtype=np.int32)


# ---------------------------------------------------------------------------
# POJ-104
# ---------------------------------------------------------------------------

def run_poj104(cfg: FinetuneConfig, tokenizer: PreTrainedTokenizerFast, out: Path) -> dict[str, Any]:
    train_rows, valid_rows, test_rows = load_poj_splits(cfg)
    encoder_params, encoder_cfg = load_pretrained_encoder_params(Path(cfg.checkpoint))
    model = EmbeddingModel(cfg=encoder_cfg, use_hidden=(cfg.embedding == "h"))

    rng = jax.random.PRNGKey(cfg.seed)
    dummy = jnp.zeros((1, cfg.max_len), dtype=jnp.int32)
    fresh = model.init(rng, dummy, deterministic=True)["params"]
    params = graft_encoder(fresh, encoder_params)

    triplets = cfg.max_train_examples or len(train_rows)
    steps_per_epoch = max(1, math.ceil(triplets / cfg.batch_size))
    total_steps = steps_per_epoch * cfg.epochs
    tx = build_optimizer(params, cfg, total_steps)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @jax.jit
    def train_step(state: TrainState, tokens: jnp.ndarray) -> tuple[TrainState, dict[str, jnp.ndarray]]:
        def loss_fn(p: Any) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
            emb = model.apply({"params": p}, tokens, deterministic=True)
            za, zp, zn = jnp.split(emb, 3, axis=0)
            za = l2_normalize(za)
            zp = l2_normalize(zp)
            zn = l2_normalize(zn)
            sim_pos = jnp.sum(za * zp, axis=-1)
            sim_neg = jnp.sum(za * zn, axis=-1)
            loss = jnp.mean(jnp.maximum(0.0, cfg.margin + sim_neg - sim_pos))
            return loss, {"loss": loss, "triplet_acc": jnp.mean(sim_pos > sim_neg)}

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        return state.apply_gradients(grads=grads), metrics

    @jax.jit
    def embed_step(params: Any, tokens: jnp.ndarray) -> jnp.ndarray:
        return l2_normalize(model.apply({"params": params}, tokens, deterministic=True))

    started = time.time()
    best_valid = -1.0
    for epoch in range(1, cfg.epochs + 1):
        sampler = PojTripletSampler(train_rows, seed=cfg.seed + epoch)
        losses: list[float] = []
        accs: list[float] = []
        for step in range(1, steps_per_epoch + 1):
            anchors, positives, negatives = sampler.next_batch(cfg.batch_size)
            tokens = jnp.asarray(tokenize_fixed(tokenizer, anchors + positives + negatives, cfg.max_len))
            state, metrics = train_step(state, tokens)
            losses.append(float(metrics["loss"]))
            accs.append(float(metrics["triplet_acc"]))
            if step == 1 or step % cfg.log_every == 0:
                log(out, {"event": "train_batch", "epoch": epoch, "step": step,
                          "loss": losses[-1], "triplet_acc": accs[-1]})
        valid = evaluate_poj_mapr(embed_step, state.params, tokenizer, valid_rows, cfg)
        best_valid = max(best_valid, valid["map_at_r"])
        log(out, {"event": "epoch", "epoch": epoch, "elapsed_s": round(time.time() - started, 2),
                  "train_loss": float(np.mean(losses)), "train_triplet_acc": float(np.mean(accs)),
                  "valid_map_at_r": valid["map_at_r"]})

    test = evaluate_poj_mapr(embed_step, state.params, tokenizer, test_rows, cfg)
    save_finetuned(out, state.params, cfg, encoder_cfg)
    return {
        "benchmark": "poj104",
        "embedding": cfg.embedding,
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "test_rows": len(test_rows),
        "best_valid_map_at_r": best_valid,
        "test": test,
    }


def evaluate_poj_mapr(
    embed_step: Any, params: Any, tokenizer: PreTrainedTokenizerFast,
    rows: list[dict[str, Any]], cfg: FinetuneConfig,
) -> dict[str, float]:
    labels = np.asarray([str(row["label"]) for row in rows])
    embeddings = encode_all(embed_step, params, tokenizer,
                            [str(row["code"]) for row in rows], cfg)
    counts = Counter(labels)
    aps: list[float] = []
    query_batch = max(1, min(cfg.eval_batch_size, 256))
    for start in range(0, len(rows), query_batch):
        end = min(start + query_batch, len(rows))
        scores = embeddings[start:end] @ embeddings.T
        for offset in range(end - start):
            scores[offset, start + offset] = -np.inf
        order = np.argsort(-scores, axis=1)
        for offset, ranked in enumerate(order):
            label = labels[start + offset]
            r = counts[label] - 1
            if r <= 0:
                continue
            top_r = ranked[:r]
            relevant = (labels[top_r] == label).astype(np.float32)
            precision = np.cumsum(relevant) / (np.arange(r) + 1)
            aps.append(float((precision * relevant).sum() / r))
    return {"map_at_r": float(np.mean(aps)) if aps else 0.0,
            "queries": float(len(aps)), "rows": float(len(rows))}


def encode_all(
    embed_step: Any, params: Any, tokenizer: PreTrainedTokenizerFast,
    texts: list[str], cfg: FinetuneConfig,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    bs = cfg.eval_batch_size
    for start in range(0, len(texts), bs):
        chunk = texts[start:start + bs]
        pad = bs - len(chunk)
        if pad:
            chunk = chunk + [chunk[-1]] * pad
        tokens = jnp.asarray(tokenize_fixed(tokenizer, chunk, cfg.max_len))
        emb = np.asarray(embed_step(params, tokens))
        outputs.append(emb[: bs - pad] if pad else emb)
    return np.concatenate(outputs, axis=0)


class PojTripletSampler:
    def __init__(self, rows: list[dict[str, Any]], *, seed: int) -> None:
        self.rows = rows
        self.rng = random.Random(seed)
        self.by_label: dict[str, list[int]] = defaultdict(list)
        for idx, row in enumerate(rows):
            self.by_label[str(row["label"])].append(idx)
        self.labels = [label for label, values in self.by_label.items() if len(values) > 1]
        if len(self.labels) < 2:
            raise RuntimeError("POJ-104 training requires at least two labels with >=2 examples")

    def next_batch(self, batch_size: int) -> tuple[list[str], list[str], list[str]]:
        anchors: list[str] = []
        positives: list[str] = []
        negatives: list[str] = []
        for _ in range(batch_size):
            label = self.rng.choice(self.labels)
            anchor_idx = self.rng.choice(self.by_label[label])
            positive_idx = self.rng.choice(self.by_label[label])
            while positive_idx == anchor_idx:
                positive_idx = self.rng.choice(self.by_label[label])
            negative_label = self.rng.choice(self.labels)
            while negative_label == label:
                negative_label = self.rng.choice(self.labels)
            negative_idx = self.rng.choice(self.by_label[negative_label])
            anchors.append(str(self.rows[anchor_idx]["code"]))
            positives.append(str(self.rows[positive_idx]["code"]))
            negatives.append(str(self.rows[negative_idx]["code"]))
        return anchors, positives, negatives


# ---------------------------------------------------------------------------
# BigCloneBench
# ---------------------------------------------------------------------------

def run_bigclonebench(
    cfg: FinetuneConfig, tokenizer: PreTrainedTokenizerFast, out: Path
) -> dict[str, Any]:
    base = Path(cfg.benchmark_dir)
    funcs = load_bigclonebench_functions(resolve_benchmark_file(base, "data.jsonl"))
    train_pairs = load_bigclonebench_pairs(resolve_benchmark_file(base, "train.txt"), funcs,
                                           max_examples=cfg.max_train_examples, seed=cfg.seed)
    valid_pairs = load_bigclonebench_pairs(resolve_benchmark_file(base, "valid.txt"), funcs,
                                           max_examples=cfg.max_valid_examples, seed=cfg.seed)
    test_pairs = load_bigclonebench_pairs(resolve_benchmark_file(base, "test.txt"), funcs,
                                          max_examples=cfg.max_test_examples, seed=cfg.seed)

    encoder_params, encoder_cfg = load_pretrained_encoder_params(Path(cfg.checkpoint))
    model = CloneClassifier(cfg=encoder_cfg, use_hidden=(cfg.embedding == "h"),
                            head_dim=encoder_cfg.hidden_size, dropout=cfg.dropout)

    rng = jax.random.PRNGKey(cfg.seed)
    rng, init_rng, drop_rng = jax.random.split(rng, 3)
    dummy = jnp.zeros((1, cfg.max_len), dtype=jnp.int32)
    fresh = model.init({"params": init_rng, "dropout": drop_rng}, dummy, dummy, deterministic=True)["params"]
    params = graft_encoder(fresh, encoder_params)

    steps_per_epoch = max(1, len(train_pairs) // cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    tx = build_optimizer(params, cfg, total_steps)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @jax.jit
    def train_step(state: TrainState, left: jnp.ndarray, right: jnp.ndarray,
                   labels: jnp.ndarray, drop_rng: jnp.ndarray) -> tuple[TrainState, jnp.ndarray]:
        def loss_fn(p: Any) -> jnp.ndarray:
            logits = model.apply({"params": p}, left, right, deterministic=False,
                                 rngs={"dropout": drop_rng})
            return optax.sigmoid_binary_cross_entropy(logits, labels).mean()

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    @jax.jit
    def score_step(params: Any, left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
        logits = model.apply({"params": params}, left, right, deterministic=True)
        return jax.nn.sigmoid(logits)

    started = time.time()
    best_valid_f1 = -1.0
    best_threshold = 0.5
    for epoch in range(1, cfg.epochs + 1):
        epoch_rng = random.Random(cfg.seed + epoch)
        epoch_rng.shuffle(train_pairs)
        losses: list[float] = []
        for step in range(1, steps_per_epoch + 1):
            batch = train_pairs[(step - 1) * cfg.batch_size: step * cfg.batch_size]
            left = jnp.asarray(tokenize_fixed(tokenizer, [p[0] for p in batch], cfg.max_len))
            right = jnp.asarray(tokenize_fixed(tokenizer, [p[1] for p in batch], cfg.max_len))
            labels = jnp.asarray([float(p[2]) for p in batch], dtype=jnp.float32)
            rng, drop = jax.random.split(rng)
            state, loss = train_step(state, left, right, labels, drop)
            losses.append(float(loss))
            if step == 1 or step % cfg.log_every == 0:
                log(out, {"event": "train_batch", "epoch": epoch, "step": step, "loss": losses[-1]})
        valid = evaluate_bigclonebench(score_step, state.params, tokenizer, valid_pairs, cfg)
        if valid["best_f1"] > best_valid_f1:
            best_valid_f1 = valid["best_f1"]
            best_threshold = valid["best_threshold"]
        log(out, {"event": "epoch", "epoch": epoch, "elapsed_s": round(time.time() - started, 2),
                  "train_loss": float(np.mean(losses)) if losses else 0.0,
                  "valid_best_f1": valid["best_f1"], "valid_roc_auc": valid["roc_auc"]})

    test = evaluate_bigclonebench(score_step, state.params, tokenizer, test_pairs, cfg,
                                  threshold=best_threshold)
    save_finetuned(out, state.params, cfg, encoder_cfg)
    return {
        "benchmark": "bigclonebench",
        "embedding": cfg.embedding,
        "train_pairs": len(train_pairs),
        "valid_pairs": len(valid_pairs),
        "test_pairs": len(test_pairs),
        "best_valid_f1": best_valid_f1,
        "valid_selected_threshold": best_threshold,
        "test": test,
    }


def evaluate_bigclonebench(
    score_step: Any, params: Any, tokenizer: PreTrainedTokenizerFast,
    pairs: list[tuple[str, str, int]], cfg: FinetuneConfig, *, threshold: float | None = None,
) -> dict[str, float]:
    scores: list[float] = []
    labels: list[int] = []
    bs = cfg.eval_batch_size
    for start in range(0, len(pairs), bs):
        chunk = pairs[start:start + bs]
        pad = bs - len(chunk)
        padded = chunk + [chunk[-1]] * pad if pad else chunk
        left = jnp.asarray(tokenize_fixed(tokenizer, [p[0] for p in padded], cfg.max_len))
        right = jnp.asarray(tokenize_fixed(tokenizer, [p[1] for p in padded], cfg.max_len))
        probs = np.asarray(score_step(params, left, right))
        if pad:
            probs = probs[: bs - pad]
        scores.extend(float(x) for x in probs)
        labels.extend(int(p[2]) for p in chunk)
    return binary_metrics(np.asarray(labels, dtype=np.int64), np.asarray(scores), threshold)


# ---------------------------------------------------------------------------
# Data loaders (mirror scripts/finetune_clone_benchmarks.py exactly)
# ---------------------------------------------------------------------------

def load_poj_splits(cfg: FinetuneConfig) -> tuple[list[dict], list[dict], list[dict]]:
    base = Path(cfg.benchmark_dir)
    train, valid, test = base / "train.jsonl", base / "valid.jsonl", base / "test.jsonl"
    if train.exists() and valid.exists() and test.exists():
        return (load_poj_rows(train, 0),
                load_poj_rows(valid, cfg.max_valid_examples),
                load_poj_rows(test, cfg.max_test_examples))
    from datasets import load_dataset  # type: ignore
    ds = load_dataset("google/code_x_glue_cc_clone_detection_poj104")

    def hf_rows(split: str, cap: int) -> list[dict]:
        rows = [{"code": str(r["code"]), "label": str(r["label"])} for r in ds[split]]
        return rows[:cap] if cap > 0 else rows

    return hf_rows("train", 0), hf_rows("validation", cfg.max_valid_examples), hf_rows("test", cfg.max_test_examples)


def load_poj_rows(path: Path, max_examples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("code") is None or row.get("label") is None:
                continue
            rows.append(row)
            if max_examples > 0 and len(rows) >= max_examples:
                break
    return rows


def load_bigclonebench_functions(path: Path) -> dict[str, str]:
    funcs: dict[str, str] = {}
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            funcs[str(row["idx"])] = str(row["func"])
    return funcs


def load_bigclonebench_pairs(
    path: Path, funcs: dict[str, str], *, max_examples: int, seed: int,
) -> list[tuple[str, str, int]]:
    pairs: list[tuple[str, str, int]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            left, right, label = parts
            if left not in funcs or right not in funcs:
                continue
            pairs.append((funcs[left], funcs[right], int(label)))
    if max_examples > 0 and len(pairs) > max_examples:
        rng = random.Random(seed)
        rng.shuffle(pairs)
        pairs = pairs[:max_examples]
    return pairs


def resolve_benchmark_file(base: Path, name: str) -> Path:
    for candidate in (base / name, base / "dataset" / name):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing {name} under {base}")


# ---------------------------------------------------------------------------
# Metrics (numpy; mirror the PyTorch baseline)
# ---------------------------------------------------------------------------

def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float | None = None) -> dict[str, float]:
    if len(labels) == 0:
        return {}
    best = best_f1(labels, scores)
    selected = best["threshold"] if threshold is None else threshold
    selected_metrics = f1_at_threshold(labels, scores, selected)
    return {
        "average_precision": average_precision(labels, scores),
        "roc_auc": roc_auc(labels, scores),
        "best_f1": best["f1"],
        "best_threshold": best["threshold"],
        "f1_at_selected_threshold": selected_metrics["f1"],
        "precision_at_selected_threshold": selected_metrics["precision"],
        "recall_at_selected_threshold": selected_metrics["recall"],
        "accuracy_at_selected_threshold": selected_metrics["accuracy"],
        "selected_threshold": selected,
        "positive_rate": float(labels.mean()),
    }


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    positives = sorted_labels.sum()
    if positives == 0:
        return 0.0
    precision = np.cumsum(sorted_labels) / (np.arange(len(sorted_labels)) + 1)
    return float((precision * sorted_labels).sum() / positives)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.0
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    positive_rank_sum = ranks[labels == 1].sum()
    return float((positive_rank_sum - len(positives) * (len(positives) + 1) / 2)
                 / (len(positives) * len(negatives)))


def best_f1(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    best = {"f1": 0.0, "threshold": 0.5}
    for threshold in np.unique(scores):
        metrics = f1_at_threshold(labels, scores, float(threshold))
        if metrics["f1"] > best["f1"]:
            best = {"f1": metrics["f1"], "threshold": float(threshold)}
    return best


def f1_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (scores >= threshold).astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    accuracy = (tp + tn) / max(1, len(labels))
    return {"precision": float(precision), "recall": float(recall),
            "f1": float(f1), "accuracy": float(accuracy)}


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

@jax.jit
def l2_normalize(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


def save_finetuned(out: Path, params: Any, cfg: FinetuneConfig, encoder_cfg: EncoderCfg) -> None:
    payload = {
        "format": "code-jepa-jax-finetune-v1",
        "benchmark": cfg.benchmark,
        "embedding": cfg.embedding,
        "params": serialization.to_bytes(jax.device_get(params)),
        "encoder_cfg": asdict(encoder_cfg),
        "config": asdict(cfg),
    }
    tmp = out / "finetuned.pkl.tmp"
    latest = out / "finetuned.pkl"
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(latest)


def log(out: Path, record: dict[str, Any]) -> None:
    record = dict(record)
    record.setdefault("time", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    with (out / "metrics.jsonl").open("a") as handle:
        handle.write(line + "\n")


def parse_args() -> FinetuneConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", choices=["poj104", "bigclonebench"], required=True)
    p.add_argument("--benchmark-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--embedding", choices=["z", "h"], default="z")
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--end-lr-ratio", type=float, default=0.1)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=123456)
    p.add_argument("--max-train-examples", type=int, default=0)
    p.add_argument("--max-valid-examples", type=int, default=0)
    p.add_argument("--max-test-examples", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--log-every", type=int, default=50)
    return FinetuneConfig(**vars(p.parse_args()))


def main() -> None:
    cfg = parse_args()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True) + "\n")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.tokenizer_path)
    log(out, {"event": "startup", "benchmark": cfg.benchmark, "embedding": cfg.embedding,
              "checkpoint": cfg.checkpoint, "devices": [str(d) for d in jax.devices()]})
    if cfg.benchmark == "poj104":
        report = run_poj104(cfg, tokenizer, out)
    else:
        report = run_bigclonebench(cfg, tokenizer, out)
    (out / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
