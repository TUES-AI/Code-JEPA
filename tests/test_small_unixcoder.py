from __future__ import annotations

import torch

from code_jepa.models import (
    ENCODER_ONLY,
    UNIXCODER_SPECIAL_TOKENS,
    SmallUniXcoder,
    count_parameters,
    ensure_unixcoder_special_tokens,
    unixcoder_tokenize,
)


def test_small_unixcoder_default_parameter_band() -> None:
    model = SmallUniXcoder.from_spec()

    assert 25_000_000 <= count_parameters(model) <= 30_000_000
    assert count_parameters(model) == 27_830_272
    assert model.config.hidden_size == 512
    assert model.config.num_hidden_layers == 6
    assert model.config.num_attention_heads == 8


def test_small_unixcoder_encoder_only_forward_shapes() -> None:
    model = SmallUniXcoder.from_spec(vocab_size=128, max_position_embeddings=18)
    input_ids = torch.tensor([[1, 4, 2, 20, 21, 2, 0, 0]])
    attention_mask = input_ids.ne(0).long()

    tokens, sentence = model.encode(input_ids, attention_mask)

    assert tokens.shape == (1, 8, 512)
    assert sentence.shape == (1, 512)


def test_unixcoder_tokenize_adds_encoder_mode_prefix() -> None:
    tokenizer = ToyTokenizer()
    added = ensure_unixcoder_special_tokens(tokenizer)

    batch = unixcoder_tokenize(
        tokenizer,
        ["ab"],
        mode=ENCODER_ONLY,
        max_length=8,
        padding="max_length",
        return_tensors="pt",
    )

    assert added == len(UNIXCODER_SPECIAL_TOKENS)
    assert batch["input_ids"].tolist() == [[1, tokenizer.vocab[ENCODER_ONLY], 2, 10, 11, 2, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 1, 0, 0]]


class ToyTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    unk_token_id = 3

    def __init__(self) -> None:
        self.vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "a": 10, "b": 11}

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def add_tokens(self, tokens: list[str], *, special_tokens: bool = False) -> int:
        del special_tokens
        added = 0
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = max(self.vocab.values()) + 1
                added += 1
        return added

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
        max_length: int,
    ) -> list[int]:
        del add_special_tokens, truncation
        return [self.vocab.get(ch, self.unk_token_id) for ch in text][:max_length]
