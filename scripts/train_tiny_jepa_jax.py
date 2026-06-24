#!/usr/bin/env python3
"""Tiny JAX Code-JEPA smoke trainer.

This is intentionally small: it validates that the generated anchor/positive/negative
triples can drive a JEPA-style latent prediction + hard-negative ranking objective.
It is not the final model.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq

try:
    import jax
    import jax.numpy as jnp
    import optax
    from flax import linen as nn
    from flax import serialization
    from flax.training import train_state
except Exception as exc:  # pragma: no cover - exercised on GPU box.
    raise SystemExit(
        "JAX smoke trainer requires jax, flax, and optax. "
        "Run on the GIANT GPU image or install training deps. "
        f"Import error: {type(exc).__name__}: {exc}"
    ) from exc


@dataclass(frozen=True)
class TrainConfig:
    data_root: str
    output_dir: str
    max_triples: int = 20000
    max_len: int = 256
    batch_size: int = 64
    steps: int = 500
    seed: int = 0
    d_model: int = 128
    z_dim: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    margin: float = 0.2
    rank_weight: float = 0.25
    log_every: int = 20
    save_every: int = 250


class TinyEncoder(nn.Module):
    max_len: int
    vocab_size: int = 257
    d_model: int = 128
    z_dim: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256

    @nn.compact
    def __call__(self, tokens: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        token_embed = nn.Embed(self.vocab_size, self.d_model, name="token_embed")(tokens)
        pos = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (self.max_len, self.d_model),
        )
        h = token_embed + pos[None, :, :]
        attn_mask = mask[:, None, None, :]
        for layer_idx in range(self.n_layers):
            y = nn.LayerNorm(name=f"attn_ln_{layer_idx}")(h)
            y = nn.SelfAttention(
                num_heads=self.n_heads,
                qkv_features=self.d_model,
                out_features=self.d_model,
                name=f"attn_{layer_idx}",
            )(y, mask=attn_mask)
            h = h + y
            y = nn.LayerNorm(name=f"ff_ln_{layer_idx}")(h)
            y = nn.Dense(self.d_ff, name=f"ff_in_{layer_idx}")(y)
            y = nn.gelu(y)
            y = nn.Dense(self.d_model, name=f"ff_out_{layer_idx}")(y)
            h = h + y
        h = nn.LayerNorm(name="final_ln")(h)
        mask_f = mask[..., None].astype(h.dtype)
        pooled = jnp.sum(h * mask_f, axis=1) / jnp.maximum(jnp.sum(mask_f, axis=1), 1.0)
        z = nn.Dense(self.z_dim, name="proj")(pooled)
        return l2_normalize(z)


class TinyCodeJepa(nn.Module):
    cfg: TrainConfig

    @nn.compact
    def __call__(
        self,
        anchor_tokens: jnp.ndarray,
        anchor_mask: jnp.ndarray,
        positive_tokens: jnp.ndarray,
        positive_mask: jnp.ndarray,
        negative_tokens: jnp.ndarray,
        negative_mask: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        encoder = TinyEncoder(
            max_len=self.cfg.max_len,
            d_model=self.cfg.d_model,
            z_dim=self.cfg.z_dim,
            n_layers=self.cfg.n_layers,
            n_heads=self.cfg.n_heads,
            d_ff=self.cfg.d_ff,
        )
        za = encoder(anchor_tokens, anchor_mask)
        zp = encoder(positive_tokens, positive_mask)
        zn = encoder(negative_tokens, negative_mask)
        pred = nn.Dense(self.cfg.d_model, name="pred_in")(za)
        pred = nn.gelu(pred)
        pred = nn.Dense(self.cfg.z_dim, name="pred_out")(pred)
        pred = l2_normalize(pred)
        return za, zp, zn, pred


def l2_normalize(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


class TrainState(train_state.TrainState):
    pass


def main() -> None:
    cfg = parse_args()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True) + "\n")

    print(json.dumps({"event": "startup", "devices": [str(d) for d in jax.devices()]}), flush=True)
    examples = load_examples(Path(cfg.data_root), max_triples=cfg.max_triples, seed=cfg.seed)
    print(json.dumps({"event": "loaded_examples", "count": len(examples)}), flush=True)
    if len(examples) < cfg.batch_size:
        raise SystemExit(f"Not enough examples: {len(examples)} < batch_size {cfg.batch_size}")

    arrays = tokenize_examples(examples, max_len=cfg.max_len)
    state = create_state(cfg)
    metrics_path = out / "metrics.jsonl"
    rng = np.random.default_rng(cfg.seed)
    start = time.time()

    for step in range(1, cfg.steps + 1):
        batch = sample_batch(arrays, cfg.batch_size, rng)
        state, metrics = train_step(state, batch, cfg.margin, cfg.rank_weight)
        if step == 1 or step % cfg.log_every == 0:
            record = {"event": "train", "step": step, "elapsed_s": round(time.time() - start, 2)}
            record.update({k: float(v) for k, v in metrics.items()})
            append_jsonl(metrics_path, record)
            print(json.dumps(record), flush=True)
        if step % cfg.save_every == 0:
            save_checkpoint(out, state, step)

    save_checkpoint(out, state, cfg.steps)
    print(json.dumps({"event": "done", "steps": cfg.steps, "output_dir": str(out)}), flush=True)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-triples", type=int, default=20000)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--z-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--rank-weight", type=float, default=0.25)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=250)
    return TrainConfig(**vars(parser.parse_args()))


def load_examples(data_root: Path, *, max_triples: int, seed: int) -> list[tuple[str, str, str]]:
    triple_files = sorted((data_root / "triples").rglob("*.parquet"))
    view_files = sorted((data_root / "views").rglob("*.parquet"))
    if not triple_files:
        raise FileNotFoundError(f"No triple parquet files under {data_root / 'triples'}")
    if not view_files:
        raise FileNotFoundError(f"No view parquet files under {data_root / 'views'}")

    triples: list[tuple[str, str, str]] = []
    for path in triple_files:
        table = pq.read_table(
            path,
            columns=["anchor_view_id", "positive_view_id", "negative_view_id"],
        )
        triples.extend(zip(*(col.to_pylist() for col in table.itercolumns())))
        if len(triples) >= max_triples * 2:
            break
    rng = random.Random(seed)
    rng.shuffle(triples)
    triples = triples[:max_triples]
    needed = {item for triple in triples for item in triple}

    view_code: dict[str, str] = {}
    for path in view_files:
        table = pq.read_table(path, columns=["view_id", "code"])
        ids = table.column("view_id").to_pylist()
        codes = table.column("code").to_pylist()
        for view_id, code in zip(ids, codes):
            if view_id in needed:
                view_code[view_id] = code
        if len(view_code) == len(needed):
            break

    examples = []
    missing = 0
    for anchor_id, positive_id, negative_id in triples:
        try:
            examples.append((view_code[anchor_id], view_code[positive_id], view_code[negative_id]))
        except KeyError:
            missing += 1
    if missing:
        print(json.dumps({"event": "missing_views", "count": missing}), flush=True)
    return examples


def tokenize_examples(
    examples: list[tuple[str, str, str]], *, max_len: int
) -> dict[str, np.ndarray]:
    anchors, positives, negatives = zip(*examples)
    a_tokens, a_mask = tokenize_codes(anchors, max_len=max_len)
    p_tokens, p_mask = tokenize_codes(positives, max_len=max_len)
    n_tokens, n_mask = tokenize_codes(negatives, max_len=max_len)
    return {
        "anchor_tokens": a_tokens,
        "anchor_mask": a_mask,
        "positive_tokens": p_tokens,
        "positive_mask": p_mask,
        "negative_tokens": n_tokens,
        "negative_mask": n_mask,
    }


def tokenize_codes(codes: Iterable[str], *, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    codes = list(codes)
    tokens = np.zeros((len(codes), max_len), dtype=np.int32)
    mask = np.zeros((len(codes), max_len), dtype=np.bool_)
    for i, code in enumerate(codes):
        encoded = np.frombuffer(code.encode("utf-8", errors="ignore"), dtype=np.uint8).astype(np.int32) + 1
        encoded = encoded[:max_len]
        tokens[i, : len(encoded)] = encoded
        mask[i, : len(encoded)] = True
    return tokens, mask


def create_state(cfg: TrainConfig) -> TrainState:
    model = TinyCodeJepa(cfg)
    rng = jax.random.PRNGKey(cfg.seed)
    dummy_tokens = jnp.ones((2, cfg.max_len), dtype=jnp.int32)
    dummy_mask = jnp.ones((2, cfg.max_len), dtype=jnp.bool_)
    params = model.init(rng, dummy_tokens, dummy_mask, dummy_tokens, dummy_mask, dummy_tokens, dummy_mask)[
        "params"
    ]
    tx = optax.adamw(cfg.lr, weight_decay=cfg.weight_decay)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def sample_batch(
    arrays: dict[str, np.ndarray], batch_size: int, rng: np.random.Generator
) -> dict[str, jnp.ndarray]:
    n = arrays["anchor_tokens"].shape[0]
    idx = rng.integers(0, n, size=(batch_size,))
    return {key: jnp.asarray(value[idx]) for key, value in arrays.items()}


@jax.jit
def train_step(
    state: TrainState, batch: dict[str, jnp.ndarray], margin: float, rank_weight: float
) -> tuple[TrainState, dict[str, jnp.ndarray]]:
    def loss_fn(params):
        za, zp, zn, pred = state.apply_fn(
            {"params": params},
            batch["anchor_tokens"],
            batch["anchor_mask"],
            batch["positive_tokens"],
            batch["positive_mask"],
            batch["negative_tokens"],
            batch["negative_mask"],
        )
        zp_sg = jax.lax.stop_gradient(zp)
        zn_sg = jax.lax.stop_gradient(zn)
        pred_loss = jnp.mean(jnp.square(pred - zp_sg))
        sim_pos = jnp.sum(za * zp_sg, axis=-1)
        sim_neg = jnp.sum(za * zn_sg, axis=-1)
        rank_loss = jnp.mean(jnp.maximum(0.0, margin + sim_neg - sim_pos))
        total = pred_loss + rank_weight * rank_loss
        metrics = {
            "loss": total,
            "pred_loss": pred_loss,
            "rank_loss": rank_loss,
            "sim_pos": jnp.mean(sim_pos),
            "sim_neg": jnp.mean(sim_neg),
            "sim_gap": jnp.mean(sim_pos - sim_neg),
        }
        return total, metrics

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics


def save_checkpoint(out: Path, state: TrainState, step: int) -> None:
    ckpt = out / f"checkpoint-step-{step:06d}.msgpack"
    ckpt.write_bytes(serialization.to_bytes(state.params))


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
