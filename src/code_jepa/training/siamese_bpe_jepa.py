#!/usr/bin/env python3
"""High-throughput single-GPU JAX trainer for tokenized Code-JEPA triples.

This trainer consumes shards produced by `scripts/tokenize_jepa_triples.py`:

    tokens[batch, view, token] uint16, view = anchor/positive/negative

The training loop never tokenizes code and never joins Parquet view ids. It is built
for throughput measurement and full-data training on one accelerator before adding
multi-GPU complexity.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import subprocess
import sys
import time
from collections import defaultdict
from functools import partial
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import optax
    from flax import linen as nn
    from flax import serialization
    from flax.training import train_state
except Exception as exc:  # pragma: no cover - exercised on GPU box.
    raise SystemExit(
        "JAX trainer requires jax, flax, and optax. Install with `pip install -e .[train]` "
        f"or use the Code-JEPA GPU image. Import error: {type(exc).__name__}: {exc}"
    ) from exc


@dataclass(frozen=True)
class TrainConfig:
    data_dirs: list[str]
    output_dir: str
    vocab_size: int = 16384
    pad_token_id: int = 0
    max_len: int = 256
    batch_size: int = 512
    steps: int = 1_000_000
    duration_minutes: float = 30.0
    seed: int = 0
    model_size: str = "roberta_25m"
    hidden_size: int = 512
    projection_dim: int = 512
    layers: int = 6
    heads: int = 8
    intermediate_size: int = 2048
    dropout: float = 0.0
    lr: float = 3e-4
    end_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    warmup_steps: int = 200
    grad_clip: float = 1.0
    margin: float = 0.2
    pos_weight: float = 1.0
    rank_weight: float = 1.0
    inbatch_weight: float = 0.1
    temperature: float = 0.05
    sigreg_weight: float = 0.05
    sigreg_slices: int = 64
    precision: str = "bf16"
    log_every: int = 20
    eval_every: int = 500
    eval_batches: int = 20
    save_every: int = 1000
    s3_sync_every: int = 1000
    s3_output_prefix: str = ""
    resume: str = ""
    dry_run_steps: int = 0
    loader_prefetch: int = 1
    target_epochs: float = 1.0


class TrainState(train_state.TrainState):
    pass


class RMSNorm(nn.Module):
    dtype: Any
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],), jnp.float32)
        y = x.astype(jnp.float32)
        y = y * jax.lax.rsqrt(jnp.mean(jnp.square(y), axis=-1, keepdims=True) + self.epsilon)
        return (y * scale).astype(self.dtype)


class CudnnSelfAttention(nn.Module):
    hidden_size: int
    heads: int
    dropout: float
    dtype: Any

    @nn.compact
    def __call__(self, x: jnp.ndarray, valid_tokens: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
        if self.hidden_size % self.heads != 0:
            raise ValueError(f"hidden_size {self.hidden_size} must be divisible by heads {self.heads}")
        head_dim = self.hidden_size // self.heads
        lengths = jnp.maximum(jnp.sum(valid_tokens, axis=1).astype(jnp.int32), 1)
        qkv = nn.DenseGeneral(
            features=(3, self.heads, head_dim),
            axis=-1,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            name="qkv",
        )(x)
        query = qkv[:, :, 0, :, :]
        key = qkv[:, :, 1, :, :]
        value = qkv[:, :, 2, :, :]
        implementation = "cudnn" if jax.default_backend() == "gpu" else None
        y = jax.nn.dot_product_attention(
            query,
            key,
            value,
            query_seq_lengths=lengths,
            key_value_seq_lengths=lengths,
            implementation=implementation,
        )
        y = nn.DenseGeneral(
            features=self.hidden_size,
            axis=(-2, -1),
            dtype=self.dtype,
            param_dtype=jnp.float32,
            name="out",
        )(y)
        return nn.Dropout(rate=self.dropout)(y, deterministic=deterministic)


class EncoderLayer(nn.Module):
    hidden_size: int
    heads: int
    intermediate_size: int
    dropout: float
    dtype: Any

    @nn.compact
    def __call__(self, x: jnp.ndarray, valid_tokens: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
        y = RMSNorm(dtype=self.dtype, name="attn_norm")(x)
        y = CudnnSelfAttention(
            hidden_size=self.hidden_size,
            heads=self.heads,
            dropout=self.dropout,
            dtype=self.dtype,
            name="attn",
        )(y, valid_tokens, deterministic=deterministic)
        x = x + y.astype(x.dtype)
        y = RMSNorm(dtype=self.dtype, name="ffn_norm")(x)
        y = nn.Dense(self.intermediate_size, dtype=self.dtype, param_dtype=jnp.float32)(y)
        y = nn.gelu(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic=deterministic)
        y = nn.Dense(self.hidden_size, dtype=self.dtype, param_dtype=jnp.float32)(y)
        return x + y.astype(x.dtype)


class ProjectionHead(nn.Module):
    hidden_size: int
    projection_dim: int
    dtype: Any

    @nn.compact
    def __call__(self, h: jnp.ndarray) -> jnp.ndarray:
        gate_value = nn.Dense(
            self.hidden_size * 8,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            name="gate_value",
        )(h)
        gate, value = jnp.split(gate_value, 2, axis=-1)
        z = jax.nn.swish(gate) * value
        z = RMSNorm(dtype=self.dtype, name="norm")(z)
        z = nn.Dense(self.projection_dim, dtype=self.dtype, param_dtype=jnp.float32, name="project")(z)
        return z.astype(jnp.float32)


class PredictorHead(nn.Module):
    projection_dim: int
    dtype: Any

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        y = RMSNorm(dtype=self.dtype, name="norm_in")(z)
        y = nn.Dense(self.projection_dim * 4, dtype=self.dtype, param_dtype=jnp.float32, name="expand")(y)
        y = nn.gelu(y)
        y = nn.Dense(self.projection_dim, dtype=self.dtype, param_dtype=jnp.float32, name="project")(y)
        return y.astype(jnp.float32)


class SiameseEncoder(nn.Module):
    cfg: TrainConfig

    @nn.compact
    def __call__(self, token_ids: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
        dtype = precision_dtype(self.cfg.precision)
        mask_1d = token_ids != self.cfg.pad_token_id
        token_embed = nn.Embed(
            num_embeddings=self.cfg.vocab_size,
            features=self.cfg.hidden_size,
            dtype=dtype,
            param_dtype=jnp.float32,
            name="token_embed",
        )(token_ids)
        pos_embed = self.param(
            "position_embed",
            nn.initializers.normal(stddev=0.02),
            (self.cfg.max_len, self.cfg.hidden_size),
            jnp.float32,
        )
        x = token_embed + pos_embed[None, : token_ids.shape[1], :].astype(dtype)
        for _ in range(self.cfg.layers):
            x = EncoderLayer(
                hidden_size=self.cfg.hidden_size,
                heads=self.cfg.heads,
                intermediate_size=self.cfg.intermediate_size,
                dropout=self.cfg.dropout,
                dtype=dtype,
            )(x, mask_1d, deterministic=deterministic)
        x = RMSNorm(dtype=jnp.float32, name="final_norm")(x)
        weights = mask_1d.astype(jnp.float32)[..., None]
        h = jnp.sum(x * weights, axis=1) / jnp.maximum(jnp.sum(weights, axis=1), 1.0)
        return ProjectionHead(
            hidden_size=self.cfg.hidden_size,
            projection_dim=self.cfg.projection_dim,
            dtype=dtype,
            name="projection_head",
        )(h)


class SiameseModel(nn.Module):
    cfg: TrainConfig

    @nn.compact
    def __call__(self, tokens: jnp.ndarray, *, deterministic: bool) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        batch, views, length = tokens.shape
        flat = tokens.reshape(batch * views, length).astype(jnp.int32)
        encoded = SiameseEncoder(self.cfg)(flat, deterministic=deterministic)
        za, zp, zn = encoded.reshape(batch, views, -1).transpose(1, 0, 2)
        pred = PredictorHead(
            projection_dim=self.cfg.projection_dim,
            dtype=precision_dtype(self.cfg.precision),
            name="predictor",
        )(za)
        return pred, za, zp, zn


class TokenizedShardLoader:
    """Single-process shuffled shard loader with asynchronous next-shard preload."""

    def __init__(self, data_dirs: list[Path], *, batch_size: int, seed: int, prefetch: int = 1) -> None:
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.executor = ThreadPoolExecutor(max_workers=max(1, prefetch))
        self.shards = discover_tokenized_shards(data_dirs)
        if not self.shards:
            raise FileNotFoundError(f"no tokenized shards under {data_dirs}")
        self.shard_order = self.rng.permutation(len(self.shards)).tolist()
        self.shard_pos = 0
        self.current_shard_index: int | None = None
        self.tokens: np.ndarray | None = None
        self.example_order: np.ndarray = np.empty((0,), dtype=np.int64)
        self.example_pos = 0
        self.future: Future[tuple[int, np.ndarray]] | None = None
        self._schedule_next()
        self._activate_next()

    def next_batch(self) -> np.ndarray:
        for _ in range(len(self.shards) + 1):
            if self.tokens is None or self.example_pos + self.batch_size > len(self.example_order):
                self._activate_next()
            assert self.tokens is not None
            if len(self.example_order) < self.batch_size:
                continue
            indices = self.example_order[self.example_pos : self.example_pos + self.batch_size]
            self.example_pos += self.batch_size
            return self.tokens[indices].astype(np.int32, copy=False)
        raise ValueError(f"all tokenized shards are smaller than batch_size={self.batch_size}")

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng_state": self.rng.bit_generator.state,
            "shard_order": self.shard_order,
            "shard_pos": self.shard_pos,
            "current_shard_index": self.current_shard_index,
            "example_order": self.example_order,
            "example_pos": self.example_pos,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rng.bit_generator.state = state["rng_state"]
        self.shard_order = list(state["shard_order"])
        self.shard_pos = int(state["shard_pos"])
        self.current_shard_index = state["current_shard_index"]
        self.example_order = np.asarray(state["example_order"], dtype=np.int64)
        self.example_pos = int(state["example_pos"])
        if self.current_shard_index is not None:
            self.tokens = load_tokens(self.shards[int(self.current_shard_index)])
        self.future = None
        self._schedule_next()

    def _schedule_next(self) -> None:
        if self.future is not None:
            return
        if self.shard_pos >= len(self.shard_order):
            self.shard_order = self.rng.permutation(len(self.shards)).tolist()
            self.shard_pos = 0
        shard_index = int(self.shard_order[self.shard_pos])
        self.shard_pos += 1
        self.future = self.executor.submit(lambda i=shard_index: (i, load_tokens(self.shards[i])))

    def _activate_next(self) -> None:
        self._schedule_next()
        assert self.future is not None
        self.current_shard_index, self.tokens = self.future.result()
        self.future = None
        self.example_order = self.rng.permutation(self.tokens.shape[0])
        self.example_pos = 0
        self._schedule_next()


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dirs", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--pad-token-id", type=int, default=0)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--duration-minutes", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-size", choices=["roberta_20m", "roberta_25m", "roberta_30m", "custom"], default="roberta_25m")
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--projection-dim", type=int, default=512)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--intermediate-size", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--end-lr-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--pos-weight", type=float, default=1.0)
    p.add_argument("--rank-weight", type=float, default=1.0)
    p.add_argument("--inbatch-weight", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--sigreg-weight", type=float, default=0.05)
    p.add_argument("--sigreg-slices", type=int, default=64)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--s3-sync-every", type=int, default=1000)
    p.add_argument("--s3-output-prefix", default="")
    p.add_argument("--resume", default="")
    p.add_argument("--dry-run-steps", type=int, default=0)
    p.add_argument("--loader-prefetch", type=int, default=1)
    p.add_argument("--target-epochs", type=float, default=1.0)
    return apply_model_preset(TrainConfig(**vars(p.parse_args())))


MODEL_PRESETS: dict[str, dict[str, int]] = {
    # Total trainable parameters include embeddings, encoder, projection head, and predictor.
    "roberta_20m": {"hidden_size": 384, "projection_dim": 384, "layers": 6, "heads": 6, "intermediate_size": 1536},
    "roberta_25m": {"hidden_size": 448, "projection_dim": 448, "layers": 6, "heads": 7, "intermediate_size": 1792},
    "roberta_30m": {"hidden_size": 512, "projection_dim": 512, "layers": 6, "heads": 8, "intermediate_size": 2048},
}


def apply_model_preset(cfg: TrainConfig) -> TrainConfig:
    if cfg.model_size == "custom":
        return cfg
    values = MODEL_PRESETS[cfg.model_size]
    return replace(cfg, **values)


def main() -> None:
    cfg = parse_args()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    out = Path(cfg.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True) + "\n")

    manifests = [load_manifest(Path(path)) for path in cfg.data_dirs]
    for manifest in [item for item in manifests if item]:
        manifest_max_len = int(manifest.get("max_len", cfg.max_len))
        if manifest_max_len > cfg.max_len:
            raise ValueError(f"manifest max_len {manifest_max_len} exceeds --max-len {cfg.max_len}")
        if int(manifest.get("vocab_size", cfg.vocab_size)) != cfg.vocab_size:
            raise ValueError(f"manifest vocab_size {manifest.get('vocab_size')} != --vocab-size {cfg.vocab_size}")

    model = SiameseModel(cfg)
    state = create_state(model, cfg)
    param_count = tree_param_count(state.params)
    total_examples = manifest_example_count([Path(d) for d in cfg.data_dirs])
    rng = jax.random.PRNGKey(cfg.seed)
    train_loader = TokenizedShardLoader([Path(d) for d in cfg.data_dirs], batch_size=cfg.batch_size, seed=cfg.seed, prefetch=cfg.loader_prefetch)
    eval_loader = TokenizedShardLoader([Path(d) for d in cfg.data_dirs], batch_size=cfg.batch_size, seed=cfg.seed + 17, prefetch=cfg.loader_prefetch)
    start_step = 0
    if cfg.resume:
        state, rng, start_step = load_checkpoint(resolve_checkpoint_path(cfg.resume, out), state, train_loader, eval_loader)
        log(out, {"event": "resumed", "step": start_step, "checkpoint": cfg.resume})

    log(
        out,
        {
            "event": "startup",
            "device": str(jax.devices()[0]),
            "shards": len(train_loader.shards),
            "tokenized_examples": total_examples,
            "parameters": param_count,
            "args": asdict(cfg),
        },
    )
    started = time.time()
    deadline = started + cfg.duration_minutes * 60 if cfg.duration_minutes > 0 else math.inf
    step = start_step
    for step in range(start_step + 1, cfg.steps + 1):
        if time.time() >= deadline:
            log(out, {"event": "deadline", "step": step})
            break
        batch_started = time.time()
        batch = train_loader.next_batch()
        batch_token_count = int(batch.shape[0] * batch.shape[1] * batch.shape[2])
        rng, step_rng = jax.random.split(rng)
        state, metrics = train_step(
            state,
            jax.device_put(batch),
            step_rng,
            cfg.margin,
            cfg.pos_weight,
            cfg.rank_weight,
            cfg.inbatch_weight,
            cfg.temperature,
            cfg.sigreg_weight,
            cfg.sigreg_slices,
        )
        jax.block_until_ready(metrics["loss"])
        elapsed = time.time() - started
        batch_s = time.time() - batch_started
        if step == 1 or step % cfg.log_every == 0:
            record = metrics_to_record(metrics)
            record.update(
                {
                    "event": "train",
                    "step": step,
                    "elapsed_s": round(elapsed, 2),
                    "batch_s": round(batch_s, 4),
                    "examples_per_s": round(cfg.batch_size / max(batch_s, 1e-9), 2),
                    "seq_len": int(batch.shape[-1]),
                    "tokens_per_s": round(batch_token_count / max(batch_s, 1e-9), 2),
                    "est_hours_per_epoch": estimate_hours(total_examples, cfg.batch_size / max(batch_s, 1e-9)),
                    "est_hours_target_epochs": estimate_hours(int(total_examples * cfg.target_epochs), cfg.batch_size / max(batch_s, 1e-9)),
                }
            )
            log(out, record)
        if cfg.eval_every > 0 and step % cfg.eval_every == 0:
            log(out, {"event": "eval", "step": step, **evaluate(state, eval_loader, cfg)})
        if step % cfg.save_every == 0:
            save_checkpoint(out, state, rng, train_loader, eval_loader, cfg, step)
        if cfg.s3_output_prefix and cfg.s3_sync_every > 0 and step % cfg.s3_sync_every == 0:
            sync_s3(out, cfg.s3_output_prefix)
        if cfg.dry_run_steps and step >= cfg.dry_run_steps:
            log(out, {"event": "dry_run_done", "step": step})
            break

    save_checkpoint(out, state, rng, train_loader, eval_loader, cfg, step)
    if cfg.s3_output_prefix:
        sync_s3(out, cfg.s3_output_prefix)
    log(out, {"event": "done", "step": step, "elapsed_s": round(time.time() - started, 2)})


def create_state(model: SiameseModel, cfg: TrainConfig) -> TrainState:
    dummy = jnp.zeros((cfg.batch_size, 3, cfg.max_len), dtype=jnp.int32)
    variables = model.init(jax.random.PRNGKey(cfg.seed), dummy, deterministic=True)
    decay_steps = max(cfg.steps, cfg.warmup_steps + 1)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.lr,
        warmup_steps=cfg.warmup_steps,
        decay_steps=decay_steps,
        end_value=cfg.lr * cfg.end_lr_ratio,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=cfg.weight_decay),
    )
    return TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx)


@jax.jit
def l2_normalize(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


def make_loss_fn(
    *,
    margin: float,
    pos_weight: float,
    rank_weight: float,
    inbatch_weight: float,
    temperature: float,
    sigreg_weight: float,
    sigreg_slices: int,
):
    def loss_fn(params: Any, state: TrainState, batch: jnp.ndarray, rng: jnp.ndarray):
        pred, za, zp, zn = state.apply_fn({"params": params}, batch, deterministic=True)
        pred_n = l2_normalize(pred)
        zp_n = l2_normalize(zp)
        zn_n = l2_normalize(zn)
        sim_pos = jnp.sum(pred_n * zp_n, axis=-1)
        sim_neg = jnp.sum(pred_n * zn_n, axis=-1)
        jepa_loss = jnp.mean(jnp.square(pred - zp))
        rank_loss = jnp.mean(jnp.maximum(0.0, margin + sim_neg - sim_pos))
        logits = pred_n @ zp_n.T / temperature
        labels = jnp.arange(batch.shape[0])
        inbatch_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        sigreg_loss = sliced_sigreg(jnp.concatenate([za, zp, zn], axis=0), rng, sigreg_slices)
        loss = pos_weight * jepa_loss + rank_weight * rank_loss + inbatch_weight * inbatch_loss + sigreg_weight * sigreg_loss
        metrics = {
            "loss": loss,
            "jepa_loss": jepa_loss,
            "rank_loss": rank_loss,
            "inbatch_loss": inbatch_loss,
            "sigreg_loss": sigreg_loss,
            "rank_acc": jnp.mean(sim_pos > sim_neg),
            "sim_pos": jnp.mean(sim_pos),
            "sim_neg": jnp.mean(sim_neg),
            "sim_gap": jnp.mean(sim_pos - sim_neg),
        }
        return loss, metrics
    return loss_fn


@partial(jax.jit, static_argnames=("sigreg_slices",), donate_argnums=(0,))
def train_step(
    state: TrainState,
    batch: jnp.ndarray,
    rng: jnp.ndarray,
    margin: float,
    pos_weight: float,
    rank_weight: float,
    inbatch_weight: float,
    temperature: float,
    sigreg_weight: float,
    sigreg_slices: int,
) -> tuple[TrainState, dict[str, jnp.ndarray]]:
    loss_fn = make_loss_fn(
        margin=margin,
        pos_weight=pos_weight,
        rank_weight=rank_weight,
        inbatch_weight=inbatch_weight,
        temperature=temperature,
        sigreg_weight=sigreg_weight,
        sigreg_slices=sigreg_slices,
    )
    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, state, batch, rng)
    return state.apply_gradients(grads=grads), metrics


@jax.jit
def eval_step(state: TrainState, batch: jnp.ndarray) -> dict[str, jnp.ndarray]:
    pred, _za, zp, zn = state.apply_fn({"params": state.params}, batch, deterministic=True)
    pred = l2_normalize(pred)
    zp = l2_normalize(zp)
    zn = l2_normalize(zn)
    sim_pos = jnp.sum(pred * zp, axis=-1)
    sim_neg = jnp.sum(pred * zn, axis=-1)
    return {
        "eval_rank_acc": jnp.mean(sim_pos > sim_neg),
        "eval_sim_pos": jnp.mean(sim_pos),
        "eval_sim_neg": jnp.mean(sim_neg),
        "eval_sim_gap": jnp.mean(sim_pos - sim_neg),
    }


def evaluate(state: TrainState, loader: TokenizedShardLoader, cfg: TrainConfig) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(cfg.eval_batches):
        batch = jax.device_put(loader.next_batch())
        metrics = eval_step(state, batch)
        for key, value in metrics.items():
            values[key].append(float(value))
    return {key: float(np.mean(item)) for key, item in values.items()}


def sliced_sigreg(samples: jnp.ndarray, rng: jnp.ndarray, num_slices: int) -> jnp.ndarray:
    directions = jax.random.normal(rng, (samples.shape[-1], num_slices), dtype=samples.dtype)
    directions = directions / jnp.maximum(jnp.linalg.norm(directions, axis=0, keepdims=True), 1e-6)
    projections = samples @ directions
    mean = jnp.mean(projections, axis=0)
    var = jnp.var(projections, axis=0)
    return jnp.mean(jnp.square(mean) + jnp.square(var - 1.0))


def discover_tokenized_shards(data_dirs: list[Path]) -> list[Path]:
    shards: list[Path] = []
    for data_dir in data_dirs:
        manifest = load_manifest(data_dir)
        if manifest:
            for item in manifest.get("shards", []):
                path = data_dir / item["path"]
                if path.exists():
                    shards.append(path)
        else:
            shards.extend(sorted(data_dir.glob("shard-*.npz")))
    return shards


def load_manifest(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def manifest_example_count(data_dirs: list[Path]) -> int:
    total = 0
    for data_dir in data_dirs:
        manifest = load_manifest(data_dir)
        if manifest:
            total += int(manifest.get("counts", {}).get("written_examples", 0))
        else:
            for shard in data_dir.glob("shard-*.npz"):
                with np.load(shard) as data:
                    total += int(data["tokens"].shape[0])
    return total


def tree_param_count(params: Any) -> int:
    return int(sum(np.prod(value.shape) for value in jax.tree_util.tree_leaves(params)))


def estimate_hours(total_examples: int, examples_per_s: float) -> float | None:
    if total_examples <= 0 or examples_per_s <= 0:
        return None
    return round(total_examples / examples_per_s / 3600, 3)


def load_tokens(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return np.asarray(data["tokens"], dtype=np.uint16)


def save_checkpoint(
    out: Path,
    state: TrainState,
    rng: jnp.ndarray,
    train_loader: TokenizedShardLoader,
    eval_loader: TokenizedShardLoader,
    cfg: TrainConfig,
    step: int,
) -> None:
    payload = {
        "format": "code-jepa-jax-train-state-v1",
        "step": step,
        "state": serialization.to_bytes(state),
        "rng": np.asarray(rng),
        "train_loader": train_loader.state_dict(),
        "eval_loader": eval_loader.state_dict(),
        "config": asdict(cfg),
    }
    tmp = out / "latest.pkl.tmp"
    latest = out / "latest.pkl"
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(latest)
    if step and step % (cfg.save_every * 5) == 0:
        checkpoint = out / f"checkpoint-step-{step:08d}.pkl"
        checkpoint.write_bytes(latest.read_bytes())


def load_checkpoint(
    path: Path,
    target_state: TrainState,
    train_loader: TokenizedShardLoader,
    eval_loader: TokenizedShardLoader,
) -> tuple[TrainState, jnp.ndarray, int]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    state = serialization.from_bytes(target_state, payload["state"])
    train_loader.load_state_dict(payload["train_loader"])
    eval_loader.load_state_dict(payload["eval_loader"])
    return state, jnp.asarray(payload["rng"]), int(payload["step"])


def resolve_checkpoint_path(value: str, out: Path) -> Path:
    if value == "latest":
        return out / "latest.pkl"
    return Path(value).expanduser().resolve()


def precision_dtype(name: str) -> Any:
    if name == "bf16":
        return jnp.bfloat16
    return jnp.float32


def metrics_to_record(metrics: dict[str, jnp.ndarray]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items()}


def sync_s3(out: Path, prefix: str) -> None:
    subprocess.run(["s5cmd", "cp", f"{out}/*", f"{prefix.rstrip('/')}/"], check=True)


def log(out: Path, record: dict[str, Any]) -> None:
    record = dict(record)
    record.setdefault("time", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    with (out / "metrics.jsonl").open("a") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    main()
