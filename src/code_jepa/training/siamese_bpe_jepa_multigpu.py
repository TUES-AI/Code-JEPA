#!/usr/bin/env python3
"""Multi-device JAX trainer for no-predictor Siamese Code-JEPA.

Consumes recursive tokenized caches produced by `scripts/tokenize_dataset_segments.py`.
The objective is intentionally pure Siamese:

    anchor   -> shared encoder -> projection -> z_anchor
    positive -> shared encoder -> projection -> z_pos
    negative -> shared encoder -> projection -> z_neg

No target encoder, no EMA, no stop-grad, and no unconditioned predictor. The
predictor variant is now an ablation, not the default paper path.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import subprocess
import time
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
except Exception as exc:  # pragma: no cover - exercised on GPU boxes.
    raise SystemExit(
        "Multi-GPU trainer requires jax, flax, and optax. Use the Code-JEPA GPU image. "
        f"Import error: {type(exc).__name__}: {exc}"
    ) from exc

from code_jepa.training.siamese_bpe_jepa import (
    MODEL_PRESETS,
    SiameseEncoder,
    estimate_hours,
    load_manifest,
    precision_dtype,
    tree_param_count,
)


BUCKET_BATCH_PRESETS: dict[str, dict[int, int]] = {
    # Per-device examples. H100 values are intentionally close to the measured
    # fixed-256 smoke, while long buckets are conservative until profiled.
    "h100": {128: 512, 256: 512, 512: 128, 1024: 32, 2048: 8},
    # A40 development defaults trade utilization for not OOMing during pmap smoke.
    "a40": {128: 256, 256: 256, 512: 64, 1024: 16, 2048: 4},
    "safe": {128: 128, 256: 128, 512: 32, 1024: 8, 2048: 2},
}


@dataclass(frozen=True)
class MultiGpuConfig:
    data_dirs: list[str]
    output_dir: str
    vocab_size: int = 16384
    pad_token_id: int = 0
    max_len: int = 2048
    steps: int = 1_000_000
    duration_minutes: float = 0.0
    target_epochs: float = 1.0
    stop_after_epochs: float = 0.0
    seed: int = 0
    num_devices: int = 0
    hardware_preset: str = "h100"
    bucket_batches: list[str] | None = None
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
    warmup_steps: int = 1_000
    grad_clip: float = 1.0
    margin: float = 0.2
    pos_weight: float = 1.0
    rank_weight: float = 1.0
    inbatch_weight: float = 0.1
    temperature: float = 0.05
    sigreg_weight: float = 0.05
    sigreg_slices: int = 64
    pos_loss: str = "cosine"
    precision: str = "bf16"
    log_every: int = 20
    eval_every: int = 0
    eval_batches: int = 20
    save_every: int = 1_000
    s3_sync_every: int = 1_000
    s3_output_prefix: str = ""
    resume: str = ""
    dry_run_steps: int = 0
    loader_prefetch: int = 1


class TrainState(train_state.TrainState):
    pass


class SiameseNoPredictorModel(nn.Module):
    cfg: MultiGpuConfig

    @nn.compact
    def __call__(self, tokens: jnp.ndarray, *, deterministic: bool) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        batch, views, length = tokens.shape
        flat = tokens.reshape(batch * views, length).astype(jnp.int32)
        encoded = SiameseEncoder(self.cfg)(flat, deterministic=deterministic)
        za, zp, zn = encoded.reshape(batch, views, -1).transpose(1, 0, 2)
        return za, zp, zn


class BucketShardLoader:
    """Shuffled tokenized-shard loader with bucket-specific batch sizes."""

    def __init__(
        self,
        data_dirs: list[Path],
        *,
        devices: int,
        bucket_batches: dict[int, int],
        seed: int,
        prefetch: int = 1,
    ) -> None:
        self.devices = devices
        self.bucket_batches = bucket_batches
        self.rng = np.random.default_rng(seed)
        self.executor = ThreadPoolExecutor(max_workers=max(1, prefetch))
        self.shards = discover_bucketed_shards(data_dirs)
        if not self.shards:
            raise FileNotFoundError(f"no tokenized shards under {data_dirs}")
        self.shard_order = self.rng.permutation(len(self.shards)).tolist()
        self.shard_pos = 0
        self.current_shard_index: int | None = None
        self.tokens: np.ndarray | None = None
        self.example_order = np.empty((0,), dtype=np.int64)
        self.example_pos = 0
        self.future: Future[tuple[int, np.ndarray]] | None = None
        self._schedule_next()
        self._activate_next()

    def next_batch(self) -> tuple[np.ndarray, int, int]:
        for _ in range(len(self.shards) + 1):
            if self.tokens is None:
                self._activate_next()
            assert self.tokens is not None
            seq_len = int(self.tokens.shape[-1])
            per_device_batch = batch_for_seq_len(seq_len, self.bucket_batches)
            global_batch = per_device_batch * self.devices
            if self.example_pos + global_batch > len(self.example_order):
                self._activate_next()
                continue
            indices = self.example_order[self.example_pos : self.example_pos + global_batch]
            self.example_pos += global_batch
            batch = self.tokens[indices].astype(np.int32, copy=False)
            batch = batch.reshape(self.devices, per_device_batch, 3, seq_len)
            return batch, per_device_batch, seq_len
        raise ValueError("all tokenized shards are smaller than their configured global batch")

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
            self.tokens = load_tokens(self.shards[int(self.current_shard_index)][0])
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
        self.future = self.executor.submit(lambda i=shard_index: (i, load_tokens(self.shards[i][0])))

    def _activate_next(self) -> None:
        self._schedule_next()
        assert self.future is not None
        self.current_shard_index, self.tokens = self.future.result()
        self.future = None
        self.example_order = self.rng.permutation(self.tokens.shape[0])
        self.example_pos = 0
        self._schedule_next()


class _TrainStep:
    def __init__(self, model: SiameseNoPredictorModel, cfg: MultiGpuConfig, devices: list[jax.Device]) -> None:
        self.model = model
        self.cfg = cfg
        self.devices = devices
        self.fn = jax.pmap(self._step, axis_name="data", devices=devices, donate_argnums=(0,))

    def __call__(self, state: TrainState, batch: jnp.ndarray, rng: jnp.ndarray) -> tuple[TrainState, dict[str, jnp.ndarray]]:
        return self.fn(state, batch, rng)

    def _step(self, state: TrainState, batch: jnp.ndarray, rng: jnp.ndarray) -> tuple[TrainState, dict[str, jnp.ndarray]]:
        def loss_fn(params: Any) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
            za, zp, zn = state.apply_fn({"params": params}, batch, deterministic=True)
            za_n = l2_normalize(za)
            zp_n = l2_normalize(zp)
            zn_n = l2_normalize(zn)
            sim_pos = jnp.sum(za_n * zp_n, axis=-1)
            sim_neg = jnp.sum(za_n * zn_n, axis=-1)
            if self.cfg.pos_loss == "mse":
                pos_loss = jnp.mean(jnp.square(za - zp))
            else:
                pos_loss = 1.0 - jnp.mean(sim_pos)
            rank_loss = jnp.mean(jnp.maximum(0.0, self.cfg.margin + sim_neg - sim_pos))
            gathered_zp = jax.lax.all_gather(zp_n, "data").reshape((-1, zp_n.shape[-1]))
            logits = za_n @ gathered_zp.T / self.cfg.temperature
            offset = jax.lax.axis_index("data") * batch.shape[0]
            labels = offset + jnp.arange(batch.shape[0])
            inbatch_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
            local_samples = jnp.concatenate([za, zp, zn], axis=0)
            all_samples = jax.lax.all_gather(local_samples, "data").reshape((-1, local_samples.shape[-1]))
            sigreg_loss = sliced_sigreg(all_samples, rng, self.cfg.sigreg_slices)
            loss = (
                self.cfg.pos_weight * pos_loss
                + self.cfg.rank_weight * rank_loss
                + self.cfg.inbatch_weight * inbatch_loss
                + self.cfg.sigreg_weight * sigreg_loss
            )
            return loss, {
                "loss": loss,
                "pos_loss": pos_loss,
                "rank_loss": rank_loss,
                "inbatch_loss": inbatch_loss,
                "sigreg_loss": sigreg_loss,
                "rank_acc": jnp.mean(sim_pos > sim_neg),
                "sim_pos": jnp.mean(sim_pos),
                "sim_neg": jnp.mean(sim_neg),
                "sim_gap": jnp.mean(sim_pos - sim_neg),
            }

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, "data")
        state = state.apply_gradients(grads=grads)
        metrics = jax.lax.pmean(metrics, "data")
        return state, metrics


class _EvalStep:
    def __init__(self, devices: list[jax.Device]) -> None:
        self.fn = jax.pmap(self._step, axis_name="data", devices=devices)

    def __call__(self, state: TrainState, batch: jnp.ndarray) -> dict[str, jnp.ndarray]:
        return self.fn(state, batch)

    @staticmethod
    def _step(state: TrainState, batch: jnp.ndarray) -> dict[str, jnp.ndarray]:
        za, zp, zn = state.apply_fn({"params": state.params}, batch, deterministic=True)
        za = l2_normalize(za)
        zp = l2_normalize(zp)
        zn = l2_normalize(zn)
        sim_pos = jnp.sum(za * zp, axis=-1)
        sim_neg = jnp.sum(za * zn, axis=-1)
        metrics = {
            "eval_rank_acc": jnp.mean(sim_pos > sim_neg),
            "eval_sim_pos": jnp.mean(sim_pos),
            "eval_sim_neg": jnp.mean(sim_neg),
            "eval_sim_gap": jnp.mean(sim_pos - sim_neg),
        }
        return jax.lax.pmean(metrics, "data")


def parse_args() -> MultiGpuConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dirs", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--pad-token-id", type=int, default=0)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--duration-minutes", type=float, default=0.0)
    p.add_argument("--target-epochs", type=float, default=1.0)
    p.add_argument("--stop-after-epochs", type=float, default=0.0, help="Stop after this many epochs over the tokenized cache; 0 disables epoch-based stopping")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-devices", type=int, default=0, help="0 means all local devices")
    p.add_argument("--hardware-preset", choices=["h100", "a40", "safe", "custom"], default="h100")
    p.add_argument("--bucket-batches", nargs="*", default=None, help="Per-device batches, e.g. 128:512 256:512 512:128 1024:32 2048:8")
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
    p.add_argument("--warmup-steps", type=int, default=1_000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--pos-weight", type=float, default=1.0)
    p.add_argument("--rank-weight", type=float, default=1.0)
    p.add_argument("--inbatch-weight", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--sigreg-weight", type=float, default=0.05)
    p.add_argument("--sigreg-slices", type=int, default=64)
    p.add_argument("--pos-loss", choices=["cosine", "mse"], default="cosine")
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--save-every", type=int, default=1_000)
    p.add_argument("--s3-sync-every", type=int, default=1_000)
    p.add_argument("--s3-output-prefix", default="")
    p.add_argument("--resume", default="")
    p.add_argument("--dry-run-steps", type=int, default=0)
    p.add_argument("--loader-prefetch", type=int, default=1)
    return apply_model_preset(MultiGpuConfig(**vars(p.parse_args())))


def main() -> None:
    cfg = parse_args()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    out = Path(cfg.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    devices = selected_devices(cfg.num_devices)
    bucket_batches = resolve_bucket_batches(cfg)
    validate_manifests([Path(item) for item in cfg.data_dirs], cfg)
    model = SiameseNoPredictorModel(cfg)
    state = create_state(model, cfg)
    replicated_state = replicate_for_pmap(state, devices)
    param_count = tree_param_count(state.params)
    total_examples = manifest_example_count([Path(d) for d in cfg.data_dirs])
    train_loader = BucketShardLoader([Path(d) for d in cfg.data_dirs], devices=len(devices), bucket_batches=bucket_batches, seed=cfg.seed, prefetch=cfg.loader_prefetch)
    eval_loader = BucketShardLoader([Path(d) for d in cfg.data_dirs], devices=len(devices), bucket_batches=bucket_batches, seed=cfg.seed + 17, prefetch=cfg.loader_prefetch)
    rng = jax.random.PRNGKey(cfg.seed)
    start_step = 0
    examples_seen = 0
    if cfg.resume:
        replicated_state, rng, start_step, examples_seen = load_checkpoint(resolve_checkpoint_path(cfg.resume, out), state, train_loader, eval_loader, devices)
    (out / "config.json").write_text(json.dumps(asdict(cfg) | {"bucket_batches_resolved": bucket_batches}, indent=2, sort_keys=True) + "\n")
    log(
        out,
        {
            "event": "startup",
            "devices": [str(device) for device in devices],
            "device_count": len(devices),
            "tokenized_examples": total_examples,
            "parameters": param_count,
            "bucket_batches": bucket_batches,
            "args": asdict(cfg),
        },
    )
    train_step = _TrainStep(model, cfg, devices)
    eval_step = _EvalStep(devices)
    started = time.time()
    deadline = started + cfg.duration_minutes * 60 if cfg.duration_minutes > 0 else math.inf
    step = start_step
    last_metrics: dict[str, jnp.ndarray] | None = None
    for step in range(start_step + 1, cfg.steps + 1):
        if time.time() >= deadline:
            log(out, {"event": "deadline", "step": step, "examples_seen": examples_seen})
            break
        batch_started = time.time()
        batch, per_device_batch, seq_len = train_loader.next_batch()
        global_batch = per_device_batch * len(devices)
        batch_token_count = int(global_batch * 3 * seq_len)
        rng, step_rng = jax.random.split(rng)
        rngs = replicate_for_pmap(step_rng, devices)
        replicated_state, metrics = train_step(replicated_state, batch, rngs)
        jax.block_until_ready(metrics["loss"])
        examples_seen += global_batch
        last_metrics = metrics
        batch_s = time.time() - batch_started
        elapsed = time.time() - started
        if step == 1 or step % cfg.log_every == 0:
            record = metrics_to_record(metrics)
            record.update(
                {
                    "event": "train",
                    "step": step,
                    "examples_seen": examples_seen,
                    "epoch_fraction": round(examples_seen / max(total_examples, 1), 6),
                    "elapsed_s": round(elapsed, 2),
                    "batch_s": round(batch_s, 4),
                    "seq_len": seq_len,
                    "per_device_batch": per_device_batch,
                    "global_batch": global_batch,
                    "examples_per_s": round(global_batch / max(batch_s, 1e-9), 2),
                    "tokens_per_s": round(batch_token_count / max(batch_s, 1e-9), 2),
                    "est_hours_per_epoch": estimate_hours(
                        total_examples,
                        global_batch / max(batch_s, 1e-9),
                    ),
                    "est_hours_target_epochs": estimate_hours(
                        int(total_examples * cfg.target_epochs),
                        global_batch / max(batch_s, 1e-9),
                    ),
                }
            )
            log(out, record)
        if cfg.stop_after_epochs > 0 and examples_seen >= int(total_examples * cfg.stop_after_epochs):
            log(
                out,
                {
                    "event": "target_epoch_done",
                    "step": step,
                    "examples_seen": examples_seen,
                    "stop_after_epochs": cfg.stop_after_epochs,
                },
            )
            break
        if cfg.eval_every > 0 and step % cfg.eval_every == 0:
            log(out, {"event": "eval", "step": step, **evaluate(replicated_state, eval_step, eval_loader, cfg)})
        if cfg.save_every > 0 and step % cfg.save_every == 0:
            save_checkpoint(out, replicated_state, rng, train_loader, eval_loader, cfg, step, examples_seen)
        if cfg.s3_output_prefix and cfg.s3_sync_every > 0 and step % cfg.s3_sync_every == 0:
            sync_s3(out, cfg.s3_output_prefix)
        if cfg.dry_run_steps and step >= cfg.dry_run_steps:
            log(out, {"event": "dry_run_done", "step": step, "examples_seen": examples_seen})
            break
    if last_metrics is not None and cfg.save_every > 0:
        save_checkpoint(out, replicated_state, rng, train_loader, eval_loader, cfg, step, examples_seen)
    if cfg.s3_output_prefix:
        sync_s3(out, cfg.s3_output_prefix)
    log(out, {"event": "done", "step": step, "examples_seen": examples_seen, "elapsed_s": round(time.time() - started, 2)})


def apply_model_preset(cfg: MultiGpuConfig) -> MultiGpuConfig:
    if cfg.model_size == "custom":
        return cfg
    return replace(cfg, **MODEL_PRESETS[cfg.model_size])


def selected_devices(num_devices: int) -> list[jax.Device]:
    devices = jax.devices()
    if not devices:
        raise RuntimeError("no JAX devices found")
    if num_devices > 0:
        if num_devices > len(devices):
            raise ValueError(f"requested {num_devices} devices but only {len(devices)} are visible")
        devices = devices[:num_devices]
    return list(devices)


def resolve_bucket_batches(cfg: MultiGpuConfig) -> dict[int, int]:
    if cfg.hardware_preset == "custom":
        if not cfg.bucket_batches:
            raise ValueError("--hardware-preset custom requires --bucket-batches")
        return parse_bucket_batches(cfg.bucket_batches)
    values = dict(BUCKET_BATCH_PRESETS[cfg.hardware_preset])
    if cfg.bucket_batches:
        values.update(parse_bucket_batches(cfg.bucket_batches))
    return values


def parse_bucket_batches(items: list[str]) -> dict[int, int]:
    out: dict[int, int] = {}
    for item in items:
        length, batch = item.split(":", 1)
        out[int(length)] = int(batch)
    if any(length <= 0 or batch <= 0 for length, batch in out.items()):
        raise ValueError(f"invalid bucket batch table: {items}")
    return out


def batch_for_seq_len(seq_len: int, bucket_batches: dict[int, int]) -> int:
    if seq_len in bucket_batches:
        return bucket_batches[seq_len]
    larger = sorted(length for length in bucket_batches if length >= seq_len)
    if larger:
        return bucket_batches[larger[0]]
    return bucket_batches[max(bucket_batches)]


def validate_manifests(data_dirs: list[Path], cfg: MultiGpuConfig) -> None:
    for data_dir in data_dirs:
        manifest = load_manifest(data_dir)
        if not manifest:
            continue
        manifest_max_len = int(manifest.get("max_len", cfg.max_len))
        if manifest_max_len > cfg.max_len:
            raise ValueError(f"manifest max_len {manifest_max_len} exceeds --max-len {cfg.max_len}")
        if int(manifest.get("vocab_size", cfg.vocab_size)) != cfg.vocab_size:
            raise ValueError(f"manifest vocab_size {manifest.get('vocab_size')} != --vocab-size {cfg.vocab_size}")


def create_state(model: SiameseNoPredictorModel, cfg: MultiGpuConfig) -> TrainState:
    dummy = jnp.zeros((1, 3, cfg.max_len), dtype=jnp.int32)
    variables = model.init(jax.random.PRNGKey(cfg.seed), dummy, deterministic=True)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.lr,
        warmup_steps=cfg.warmup_steps,
        decay_steps=max(cfg.steps, cfg.warmup_steps + 1),
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


def sliced_sigreg(samples: jnp.ndarray, rng: jnp.ndarray, num_slices: int) -> jnp.ndarray:
    directions = jax.random.normal(rng, (samples.shape[-1], num_slices), dtype=samples.dtype)
    directions = directions / jnp.maximum(jnp.linalg.norm(directions, axis=0, keepdims=True), 1e-6)
    projections = samples @ directions
    mean = jnp.mean(projections, axis=0)
    var = jnp.var(projections, axis=0)
    return jnp.mean(jnp.square(mean) + jnp.square(var - 1.0))


def evaluate(state: TrainState, eval_step: _EvalStep, loader: BucketShardLoader, cfg: MultiGpuConfig) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for _ in range(cfg.eval_batches):
        batch, _per_device_batch, _seq_len = loader.next_batch()
        metrics = eval_step(state, batch)
        for key, value in metrics.items():
            values.setdefault(key, []).append(float(jax.device_get(value)[0]))
    return {key: float(np.mean(item)) for key, item in values.items()}


def discover_bucketed_shards(data_dirs: list[Path]) -> list[tuple[Path, int]]:
    shards: list[tuple[Path, int]] = []
    for data_dir in data_dirs:
        manifest = load_manifest(data_dir)
        if manifest:
            for item in manifest.get("shards", []):
                path = data_dir / item["path"]
                if path.exists():
                    shards.append((path, int(item.get("max_len", 0))))
        else:
            for path in sorted(data_dir.rglob("shard-*.npz")):
                with np.load(path) as data:
                    shards.append((path, int(data["tokens"].shape[-1])))
    return shards


def load_tokens(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return np.asarray(data["tokens"], dtype=np.uint16)


def manifest_example_count(data_dirs: list[Path]) -> int:
    total = 0
    for data_dir in data_dirs:
        manifest = load_manifest(data_dir)
        if manifest:
            total += int(manifest.get("counts", {}).get("written_examples", 0))
        else:
            for shard in data_dir.rglob("shard-*.npz"):
                with np.load(shard) as data:
                    total += int(data["tokens"].shape[0])
    return total


def metrics_to_record(metrics: dict[str, jnp.ndarray]) -> dict[str, float]:
    return {key: float(jax.device_get(value)[0]) for key, value in metrics.items()}


def replicate_for_pmap(value: Any, devices: list[jax.Device]) -> Any:
    mesh = jax.sharding.Mesh(np.asarray(devices), ("data",))

    def put_leaf(leaf: Any) -> Any:
        arr = np.asarray(jax.device_get(leaf))
        stacked = np.broadcast_to(arr, (len(devices),) + arr.shape).copy()
        sharding = jax.sharding.NamedSharding(
            mesh,
            jax.sharding.PartitionSpec("data", *([None] * arr.ndim)),
        )
        return jax.device_put(stacked, sharding)

    return jax.tree_util.tree_map(put_leaf, value)


def unreplicate(value: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: jax.device_get(x[0]), value)


def save_checkpoint(
    out: Path,
    state: TrainState,
    rng: jnp.ndarray,
    train_loader: BucketShardLoader,
    eval_loader: BucketShardLoader,
    cfg: MultiGpuConfig,
    step: int,
    examples_seen: int,
) -> None:
    payload = {
        "format": "code-jepa-jax-multigpu-train-state-v1",
        "step": step,
        "examples_seen": examples_seen,
        "state": serialization.to_bytes(unreplicate(state)),
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
    if step and cfg.save_every > 0 and step % (cfg.save_every * 5) == 0:
        checkpoint = out / f"checkpoint-step-{step:08d}.pkl"
        checkpoint.write_bytes(latest.read_bytes())


def load_checkpoint(
    path: Path,
    target_state: TrainState,
    train_loader: BucketShardLoader,
    eval_loader: BucketShardLoader,
    devices: list[jax.Device],
) -> tuple[TrainState, jnp.ndarray, int, int]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    state = serialization.from_bytes(target_state, payload["state"])
    train_loader.load_state_dict(payload["train_loader"])
    eval_loader.load_state_dict(payload["eval_loader"])
    return replicate_for_pmap(state, devices), jnp.asarray(payload["rng"]), int(payload["step"]), int(payload.get("examples_seen", 0))


def resolve_checkpoint_path(value: str, out: Path) -> Path:
    if value == "latest":
        return out / "latest.pkl"
    return Path(value).expanduser().resolve()


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
