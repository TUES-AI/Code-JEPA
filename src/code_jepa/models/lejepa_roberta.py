"""Torch/RoBERTa mockup of a LeJEPA-style code encoder.

This is intentionally small: it is a local prototype for checking the shape of
Code-JEPA ideas before committing to a full training recipe.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F


def _distributed_max_step(step: torch.Tensor) -> torch.Tensor:
    """Synchronize the random slicing seed when distributed training is active."""

    if dist.is_available() and dist.is_initialized():
        synced = step.clone()
        dist.all_reduce(synced, op=dist.ReduceOp.MAX)
        return synced
    return step


def masked_mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token states with an attention mask."""

    mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
    total = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return total / denom


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE over valid token positions."""

    weights = mask.to(dtype=prediction.dtype).unsqueeze(-1)
    squared = (prediction - target).pow(2) * weights
    denom = (weights.sum() * prediction.size(-1)).clamp_min(1.0)
    return squared.sum() / denom


def flatten_masked_tokens(values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Flatten valid token vectors to a sample matrix."""

    active = attention_mask.bool()
    if active.any():
        return values[active]
    return values.reshape(-1, values.size(-1))


class ProjectionHead(nn.Module):
    """Small projection head used for semantic and local code geometry."""

    def __init__(self, input_dim: int, output_dim: int, *, normalize: bool = True) -> None:
        super().__init__()
        self.normalize = normalize
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        if self.normalize:
            z = F.normalize(z, dim=-1)
        return z


class GaussianMomentStatistic(nn.Module):
    """Cheap univariate normality statistic for sliced SIGReg-like regularization."""

    def forward(self, projected: torch.Tensor) -> torch.Tensor:
        mean = projected.mean(dim=-2)
        var = projected.var(dim=-2, unbiased=False)
        return mean.pow(2) + (var - 1.0).pow(2)


class SlicedGaussianRegularizer(nn.Module):
    """Project embeddings to random 1D slices and penalize non-Gaussian moments."""

    def __init__(
        self,
        *,
        num_slices: int = 64,
        reduction: str | None = "mean",
        clip_value: float | None = None,
        statistic: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", None}:
            raise ValueError("reduction must be 'mean', 'sum', or None")
        self.num_slices = num_slices
        self.reduction = reduction
        self.clip_value = clip_value
        self.statistic = statistic or GaussianMomentStatistic()
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))
        self._generator: torch.Generator | None = None
        self._generator_device: torch.device | None = None

    def _get_generator(self, device: torch.device, seed: int) -> torch.Generator:
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(seed)
        return self._generator

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        if samples.dim() < 2:
            raise ValueError("samples must have shape (..., num_samples, dim) or (num_samples, dim)")
        if samples.size(-2) < 2:
            return samples.new_zeros(())

        with torch.no_grad():
            synced_step = _distributed_max_step(self.global_step.clone())
            generator = self._get_generator(samples.device, int(synced_step.item()))
            directions = torch.randn(
                samples.size(-1),
                self.num_slices,
                device=samples.device,
                dtype=samples.dtype,
                generator=generator,
            )
            directions = F.normalize(directions, dim=0)
            self.global_step.add_(1)

        stats = self.statistic(samples @ directions)
        if self.clip_value is not None:
            stats = torch.where(stats < self.clip_value, stats.new_zeros(()), stats)
        if self.reduction == "mean":
            return stats.mean()
        if self.reduction == "sum":
            return stats.sum()
        return stats


@dataclass
class CodeLeJepaEmbeddings:
    """Two-head output from the shared RoBERTa code encoder."""

    last_hidden_state: torch.Tensor
    pooled: torch.Tensor
    semantic: torch.Tensor
    local: torch.Tensor
    attention_mask: torch.Tensor


@dataclass
class CodeLeJepaPairOutput:
    """Mock JEPA losses for context and target code views."""

    loss: torch.Tensor
    semantic_jepa_loss: torch.Tensor
    local_jepa_loss: torch.Tensor
    sigreg_loss: torch.Tensor
    context: CodeLeJepaEmbeddings
    target: CodeLeJepaEmbeddings
    semantic_prediction: torch.Tensor
    local_prediction: torch.Tensor


class RobertaCodeLeJepa(nn.Module):
    """RoBERTa-backed code-input LeJEPA mockup with semantic and local heads."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        projection_dim: int = 256,
        num_slices: int = 64,
        sigreg_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        hidden_size = int(encoder.config.hidden_size)
        self.projection_dim = projection_dim
        self.sigreg_weight = sigreg_weight

        self.semantic_head = ProjectionHead(hidden_size, projection_dim)
        self.local_head = ProjectionHead(hidden_size, projection_dim)
        self.semantic_predictor = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.local_predictor = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.semantic_sigreg = SlicedGaussianRegularizer(num_slices=num_slices)
        self.local_sigreg = SlicedGaussianRegularizer(num_slices=num_slices)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "roberta-base",
        *,
        projection_dim: int = 256,
        num_slices: int = 64,
        sigreg_weight: float = 0.05,
        local_files_only: bool = False,
    ) -> "RobertaCodeLeJepa":
        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        return cls(
            encoder,
            projection_dim=projection_dim,
            num_slices=num_slices,
            sigreg_weight=sigreg_weight,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> CodeLeJepaEmbeddings:
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = encoded.last_hidden_state
        pooled = masked_mean_pool(hidden, attention_mask)
        return CodeLeJepaEmbeddings(
            last_hidden_state=hidden,
            pooled=pooled,
            semantic=self.semantic_head(pooled),
            local=self.local_head(hidden),
            attention_mask=attention_mask,
        )

    def forward_pair(
        self,
        *,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
    ) -> CodeLeJepaPairOutput:
        context = self(context_input_ids, context_attention_mask)
        target = self(target_input_ids, target_attention_mask)

        semantic_prediction = self.semantic_predictor(context.semantic)
        semantic_loss = F.mse_loss(semantic_prediction, target.semantic.detach())

        local_prediction = self.local_predictor(context.local)
        seq_len = min(local_prediction.size(1), target.local.size(1))
        local_mask = context.attention_mask[:, :seq_len] * target.attention_mask[:, :seq_len]
        local_loss = masked_mse(
            local_prediction[:, :seq_len],
            target.local[:, :seq_len].detach(),
            local_mask,
        )

        semantic_samples = torch.cat([context.semantic, target.semantic], dim=0)
        local_samples = torch.cat(
            [
                flatten_masked_tokens(context.local, context.attention_mask),
                flatten_masked_tokens(target.local, target.attention_mask),
            ],
            dim=0,
        )
        sigreg_loss = self.semantic_sigreg(semantic_samples) + self.local_sigreg(local_samples)
        loss = semantic_loss + local_loss + self.sigreg_weight * sigreg_loss

        return CodeLeJepaPairOutput(
            loss=loss,
            semantic_jepa_loss=semantic_loss,
            local_jepa_loss=local_loss,
            sigreg_loss=sigreg_loss,
            context=context,
            target=target,
            semantic_prediction=semantic_prediction,
            local_prediction=local_prediction,
        )
