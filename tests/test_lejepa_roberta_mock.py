import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import RobertaConfig, RobertaModel

from code_jepa.models import RobertaCodeLeJepa, SlicedGaussianRegularizer


def tiny_model() -> RobertaCodeLeJepa:
    config = RobertaConfig(
        vocab_size=128,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        pad_token_id=1,
    )
    encoder = RobertaModel(config)
    return RobertaCodeLeJepa(encoder, projection_dim=8, num_slices=4, sigreg_weight=0.1)


def test_roberta_code_lejepa_exposes_two_heads() -> None:
    model = tiny_model()
    input_ids = torch.randint(4, 100, (2, 12))
    attention_mask = torch.ones_like(input_ids)

    out = model(input_ids=input_ids, attention_mask=attention_mask)

    assert out.semantic.shape == (2, 8)
    assert out.local.shape == (2, 12, 8)
    assert out.last_hidden_state.shape == (2, 12, 16)


def test_roberta_code_lejepa_pair_loss_is_scalar() -> None:
    model = tiny_model()
    context_ids = torch.randint(4, 100, (2, 12))
    target_ids = torch.randint(4, 100, (2, 10))

    out = model.forward_pair(
        context_input_ids=context_ids,
        context_attention_mask=torch.ones_like(context_ids),
        target_input_ids=target_ids,
        target_attention_mask=torch.ones_like(target_ids),
    )

    assert out.loss.ndim == 0
    assert out.semantic_jepa_loss.ndim == 0
    assert out.local_jepa_loss.ndim == 0
    assert out.sigreg_loss.ndim == 0
    assert out.semantic_prediction.shape == (2, 8)
    assert out.local_prediction.shape == (2, 12, 8)


def test_sliced_gaussian_regularizer_keeps_slice_dimension_when_unreduced() -> None:
    reg = SlicedGaussianRegularizer(num_slices=5, reduction=None)
    samples = torch.randn(16, 8)

    stats = reg(samples)

    assert stats.shape == (5,)
