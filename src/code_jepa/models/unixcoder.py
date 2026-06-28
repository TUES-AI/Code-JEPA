"""Small UniXcoder-compatible RoBERTa variants for Code-JEPA baselines.

The original UniXcoder wrapper uses a RoBERTa model with mode-prefix tokens and
different attention masks for encoder-only, decoder-only, and encoder-decoder
usage. Code-JEPA's small ablation tier only needs the encoder-only embedding
path, but keeping the mode-prefix convention makes this model a useful compact
UniXcoder-style baseline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase, RobertaConfig, RobertaModel

from code_jepa.models.lejepa_roberta import masked_mean_pool

UniXcoderMode = Literal["<encoder-only>", "<decoder-only>", "<encoder-decoder>"]

ENCODER_ONLY: UniXcoderMode = "<encoder-only>"
DECODER_ONLY: UniXcoderMode = "<decoder-only>"
ENCODER_DECODER: UniXcoderMode = "<encoder-decoder>"
UNIXCODER_MODE_TOKENS: tuple[UniXcoderMode, ...] = (
    ENCODER_ONLY,
    DECODER_ONLY,
    ENCODER_DECODER,
)
UNIXCODER_SPECIAL_TOKENS: tuple[str, ...] = (*UNIXCODER_MODE_TOKENS, "<mask0>")


@dataclass(frozen=True)
class SmallUniXcoderSpec:
    """Config preset for a 25-30M parameter UniXcoder-style model.

    With the default 16k vocabulary and tied LM head this is about 28M unique
    trainable parameters, matching the small Code-JEPA tier in ``Ablations.md``.
    """

    vocab_size: int = 16_384
    hidden_size: int = 512
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    max_position_embeddings: int = 514
    type_vocab_size: int = 1
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5


def small_unixcoder_config(
    spec: SmallUniXcoderSpec | None = None,
    **overrides: Any,
) -> RobertaConfig:
    """Build the default small UniXcoder-compatible ``RobertaConfig``."""

    values = asdict(spec or SmallUniXcoderSpec())
    values.update({key: value for key, value in overrides.items() if value is not None})
    return RobertaConfig(
        vocab_size=int(values["vocab_size"]),
        hidden_size=int(values["hidden_size"]),
        num_hidden_layers=int(values["num_hidden_layers"]),
        num_attention_heads=int(values["num_attention_heads"]),
        intermediate_size=int(values["intermediate_size"]),
        max_position_embeddings=int(values["max_position_embeddings"]),
        type_vocab_size=int(values["type_vocab_size"]),
        pad_token_id=int(values["pad_token_id"]),
        bos_token_id=int(values["bos_token_id"]),
        eos_token_id=int(values["eos_token_id"]),
        hidden_dropout_prob=float(values["hidden_dropout_prob"]),
        attention_probs_dropout_prob=float(values["attention_probs_dropout_prob"]),
        initializer_range=float(values["initializer_range"]),
        layer_norm_eps=float(values["layer_norm_eps"]),
        is_decoder=True,
        use_cache=True,
    )


class SmallUniXcoder(nn.Module):
    """Compact UniXcoder-style RoBERTa model with tied token LM head."""

    def __init__(self, config: RobertaConfig | None = None) -> None:
        super().__init__()
        self.config = config or small_unixcoder_config()
        self.config.is_decoder = True
        self.encoder = RobertaModel(self.config, attn_implementation="eager")
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.encoder.embeddings.word_embeddings.weight
        self.register_buffer(
            "causal_bias",
            torch.tril(
                torch.ones(
                    self.config.max_position_embeddings,
                    self.config.max_position_embeddings,
                    dtype=torch.bool,
                )
            ).view(1, self.config.max_position_embeddings, self.config.max_position_embeddings),
            persistent=False,
        )

    @classmethod
    def from_spec(
        cls,
        spec: SmallUniXcoderSpec | None = None,
        **overrides: Any,
    ) -> "SmallUniXcoder":
        """Instantiate the default small variant, optionally overriding config values."""

        return cls(small_unixcoder_config(spec, **overrides))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        mode: UniXcoderMode = ENCODER_ONLY,
        **kwargs: Any,
    ) -> Any:
        """Run the underlying RoBERTa model with a UniXcoder-style attention mask."""

        model_attention_mask = self._model_attention_mask(input_ids, attention_mask, mode)
        return self.encoder(input_ids=input_ids, attention_mask=model_attention_mask, **kwargs)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        mode: UniXcoderMode = ENCODER_ONLY,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return token embeddings and mask-aware mean pooled sentence embeddings."""

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        output = self(input_ids=input_ids, attention_mask=attention_mask, mode=mode)
        sentence_embeddings = masked_mean_pool(output.last_hidden_state, attention_mask)
        return output.last_hidden_state, sentence_embeddings

    def _model_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        mode: UniXcoderMode,
    ) -> torch.Tensor:
        if mode not in UNIXCODER_MODE_TOKENS:
            raise ValueError(f"unknown UniXcoder mode: {mode}")
        if attention_mask is not None and attention_mask.dim() == 3:
            return attention_mask

        valid = (
            attention_mask
            if attention_mask is not None
            else input_ids.ne(self.config.pad_token_id)
        )
        valid = valid.to(dtype=torch.bool, device=input_ids.device)
        if mode == DECODER_ONLY:
            seq_len = input_ids.size(-1)
            if seq_len > self.causal_bias.size(-1):
                raise ValueError(
                    f"sequence length {seq_len} exceeds max_position_embeddings "
                    f"{self.causal_bias.size(-1)}"
                )
            causal = self.causal_bias[:, :seq_len, :seq_len].to(input_ids.device)
            return causal & valid[:, None, :]
        return valid[:, None, :] & valid[:, :, None]


def ensure_unixcoder_special_tokens(tokenizer: PreTrainedTokenizerBase) -> int:
    """Add UniXcoder mode tokens to a tokenizer if they are missing."""

    vocab = tokenizer.get_vocab()
    missing = [token for token in UNIXCODER_SPECIAL_TOKENS if token not in vocab]
    if not missing:
        return 0
    return int(tokenizer.add_tokens(missing, special_tokens=True))


def unixcoder_tokenize(
    tokenizer: PreTrainedTokenizerBase,
    inputs: list[str],
    *,
    mode: UniXcoderMode = ENCODER_ONLY,
    max_length: int = 512,
    padding: bool | Literal["longest", "max_length"] = False,
    return_tensors: Literal["pt"] | None = None,
) -> dict[str, Any]:
    """Tokenize strings with UniXcoder's ``<bos> mode <eos> ...`` convention."""

    if mode not in UNIXCODER_MODE_TOKENS:
        raise ValueError(f"unknown UniXcoder mode: {mode}")
    if max_length < 4:
        raise ValueError("max_length must be at least 4 for UniXcoder mode prefixes")

    bos_token_id = _first_token_id(tokenizer, "bos_token_id", "cls_token_id")
    eos_token_id = _first_token_id(tokenizer, "eos_token_id", "sep_token_id")
    pad_token_id = _first_token_id(tokenizer, "pad_token_id")
    mode_token_id = tokenizer.convert_tokens_to_ids(mode)
    if mode_token_id is None or (
        mode_token_id == tokenizer.unk_token_id and mode not in tokenizer.get_vocab()
    ):
        raise ValueError(f"tokenizer is missing UniXcoder mode token {mode!r}")

    encoded: list[list[int]] = []
    for text in inputs:
        body_limit = _body_limit(mode, max_length)
        body = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=body_limit,
        )
        ids = [bos_token_id, mode_token_id, eos_token_id] + body
        if mode != DECODER_ONLY:
            ids.append(eos_token_id)
        encoded.append(ids)

    pad_to = _padding_length(encoded, max_length=max_length, padding=padding)
    attention_masks: list[list[int]] = []
    if pad_to is not None:
        for ids in encoded:
            pad_count = max(0, pad_to - len(ids))
            attention_masks.append([1] * len(ids) + [0] * pad_count)
            ids.extend([pad_token_id] * pad_count)
    else:
        attention_masks = [[1] * len(ids) for ids in encoded]

    if return_tensors == "pt":
        return {
            "input_ids": torch.tensor(encoded, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }
    return {"input_ids": encoded, "attention_mask": attention_masks}


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Count unique parameters, so tied LM-head weights are not double counted."""

    seen: set[int] = set()
    total = 0
    for parameter in module.parameters():
        if trainable_only and not parameter.requires_grad:
            continue
        ident = id(parameter)
        if ident in seen:
            continue
        seen.add(ident)
        total += parameter.numel()
    return total


def _body_limit(mode: UniXcoderMode, max_length: int) -> int:
    if mode == ENCODER_ONLY:
        return max(0, max_length - 4)
    if mode == DECODER_ONLY:
        return max(0, max_length - 3)
    return max(0, max_length - 5)


def _padding_length(
    encoded: list[list[int]],
    *,
    max_length: int,
    padding: bool | Literal["longest", "max_length"],
) -> int | None:
    if padding is True or padding == "max_length":
        return max_length
    if padding == "longest":
        return max(len(ids) for ids in encoded) if encoded else 0
    return None


def _first_token_id(tokenizer: PreTrainedTokenizerBase, *attrs: str) -> int:
    for attr in attrs:
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    raise ValueError(f"tokenizer is missing required token id from {attrs}")
