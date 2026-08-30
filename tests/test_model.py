import argparse
import random
from pathlib import Path

import numpy as np
import torch
from hypothesis import HealthCheck, given, settings, strategies as st

from scripts.count_params import (
    assert_matches_final_parameter_contract,
    assert_within_competition_cap,
    count_unique_trainable_parameters,
    unique_trainable_parameters,
)
from tinybench_lm import ModelConfig, TinyBenchLM
from tinybench_lm.checkpoint import restore_checkpoint_state, save_checkpoint
from tinybench_lm.config import (
    COMPETITION_PARAMETER_CAP,
    FINAL_CAP_HEADROOM,
    FINAL_CONFIG_PATH,
    FINAL_PARAMETER_COUNT,
)
from tinybench_lm.data import PackedTokenDataset


def test_final_architecture_matches_frozen_parameter_contract() -> None:
    config = ModelConfig.from_json(FINAL_CONFIG_PATH)
    assert config == ModelConfig()
    assert config.to_dict() == {
        "vocab_size": 12_288,
        "max_seq_len": 1_024,
        "n_layers": 14,
        "d_model": 512,
        "n_heads": 8,
        "n_kv_heads": 4,
        "d_ff": 1_504,
        "rope_theta": 10_000.0,
        "rms_norm_eps": 1e-5,
        "dropout": 0.0,
        "bias": False,
        "tie_embeddings": True,
    }

    model = TinyBenchLM(config)
    unique_parameters = unique_trainable_parameters(model)
    assert len(unique_parameters) == len({id(parameter) for parameter in unique_parameters})
    assert count_unique_trainable_parameters(model) == FINAL_PARAMETER_COUNT
    assert model.count_parameters() == FINAL_PARAMETER_COUNT
    assert COMPETITION_PARAMETER_CAP - FINAL_PARAMETER_COUNT == FINAL_CAP_HEADROOM == 341_632
    assert all("bias" not in name for name, _ in model.named_parameters())

    embedding_weight = model.token_embedding.weight
    output_weight = model.output_weight
    assert output_weight is embedding_weight
    assert output_weight.untyped_storage().data_ptr() == embedding_weight.untyped_storage().data_ptr()
    assert output_weight.untyped_storage().nbytes() == embedding_weight.untyped_storage().nbytes()


def test_parameter_contract_fails_above_the_competition_cap() -> None:
    import pytest

    final_config = ModelConfig.from_json(FINAL_CONFIG_PATH)
    final_count = count_unique_trainable_parameters(TinyBenchLM(final_config))
    assert assert_within_competition_cap(final_count) == FINAL_CAP_HEADROOM
    assert_matches_final_parameter_contract(final_count)

    over_cap_config = ModelConfig(
        vocab_size=65_535,
        max_seq_len=64,
        n_layers=1,
        d_model=1_024,
        n_heads=16,
        n_kv_heads=4,
        d_ff=1_024,
    )
    over_cap_count = count_unique_trainable_parameters(TinyBenchLM(over_cap_config))
    assert over_cap_count > COMPETITION_PARAMETER_CAP
    with pytest.raises(SystemExit, match="exceeds the competition parameter cap"):
        assert_within_competition_cap(over_cap_count)
    with pytest.raises(SystemExit, match="does not match the frozen parameter contract"):
        assert_matches_final_parameter_contract(over_cap_count)

    with pytest.raises(SystemExit, match="does not match the frozen parameter contract"):
        assert_matches_final_parameter_contract(FINAL_PARAMETER_COUNT - 1)


def test_legacy_baseline_is_a_compatible_final_alias() -> None:
    assert ModelConfig.from_json("configs/baseline_49m.json") == ModelConfig.from_json(
        FINAL_CONFIG_PATH
    )


def test_pilot_and_rejected_configs_remain_available_and_labeled() -> None:
    pilot_path = Path("configs/pilot_12m.json")
    rejected_path = Path("configs/deep_thin_gqa_49m.json")
    assert ModelConfig.from_json(pilot_path).vocab_size == 8_192
    assert ModelConfig.from_json(rejected_path).n_layers == 24

    catalog = Path("configs/README.md").read_text(encoding="utf-8")
    assert "pipeline and bounded-training pilot" in catalog
    assert "rejected architecture-research candidate" in catalog


@st.composite
def valid_tiny_configurations(draw: st.DrawFn) -> ModelConfig:
    d_model, n_heads, n_kv_heads = draw(
        st.sampled_from(((8, 1, 1), (16, 2, 1), (16, 4, 2), (24, 3, 1), (32, 4, 2)))
    )
    return ModelConfig(
        vocab_size=draw(st.integers(min_value=8, max_value=64)),
        max_seq_len=draw(st.integers(min_value=2, max_value=16)),
        n_layers=draw(st.integers(min_value=1, max_value=2)),
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_ff=draw(st.integers(min_value=d_model, max_value=4 * d_model)),
        dropout=draw(st.sampled_from((0.0, 0.1))),
        bias=draw(st.booleans()),
    )


# **Validates: Requirements 3.1, 3.3**
@given(config=valid_tiny_configurations())
@settings(
    max_examples=8,
    deadline=None,
    derandomize=True,
    suppress_health_check=(HealthCheck.too_slow,),
)
def test_unique_parameter_counter_preserves_generic_valid_configs(config: ModelConfig) -> None:
    model = TinyBenchLM(config)
    independently_unique = {
        id(parameter): parameter
        for _, parameter in model.named_parameters(remove_duplicate=False)
        if parameter.requires_grad
    }
    assert unique_trainable_parameters(model) == tuple(independently_unique.values())
    assert count_unique_trainable_parameters(model) == sum(
        parameter.numel() for parameter in independently_unique.values()
    )
    assert model.count_parameters() == count_unique_trainable_parameters(model)


def test_deep_thin_candidate_stays_under_parameter_cap() -> None:
    config = ModelConfig.from_json("configs/deep_thin_gqa_49m.json")
    model = TinyBenchLM(config)
    assert model.count_parameters() == 49_367_424
    assert model.count_parameters() <= 50_000_000


def test_forward_and_loss() -> None:
    config = ModelConfig(
        vocab_size=256,
        max_seq_len=32,
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
    )
    model = TinyBenchLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(input_ids, input_ids)
    assert logits.shape == (2, 16, config.vocab_size)
    assert loss is not None and torch.isfinite(loss)


def test_non_contiguous_shifted_targets() -> None:
    config = ModelConfig(
        vocab_size=256,
        max_seq_len=32,
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
    )
    model = TinyBenchLM(config)
    packed = torch.randint(0, config.vocab_size, (2, 17))
    inputs, targets = packed[:, :-1], packed[:, 1:]
    assert not targets.is_contiguous()
    _, loss = model(inputs, targets)
    assert loss is not None and torch.isfinite(loss)


def test_checkpoint_resume_restores_rng_and_sampler_state(tmp_path) -> None:
    config = ModelConfig(
        vocab_size=256,
        max_seq_len=16,
        n_layers=1,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
    )
    token_path = tmp_path / "tokens.bin"
    (np.arange(2048, dtype=np.uint16) % config.vocab_size).tofile(token_path)

    def make_training_state():
        model = TinyBenchLM(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        train_data = PackedTokenDataset(token_path, seed=17)
        validation_data = PackedTokenDataset(token_path, seed=23)
        return model, optimizer, train_data, validation_data

    def take_step(model, optimizer, dataset):
        inputs, targets = dataset.get_batch(batch_size=2, seq_len=8, device=torch.device("cpu"))
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        assert loss is not None
        loss.backward()
        optimizer.step()
        return loss.detach().clone(), inputs.detach().clone()

    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    model, optimizer, train_data, validation_data = make_training_state()
    take_step(model, optimizer, train_data)
    checkpoint_path = tmp_path / "resume.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        config,
        argparse.Namespace(test=True),
        train_data,
        validation_data,
        step=0,
        best_validation_loss=1.25,
    )

    continuous_loss, continuous_inputs = take_step(model, optimizer, train_data)
    continuous_parameters = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    continuous_random = (random.random(), np.random.random(), torch.rand(()))
    continuous_validation = validation_data.get_batch(1, 8, torch.device("cpu"))[0]

    resumed_model, resumed_optimizer, resumed_train, resumed_validation = make_training_state()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    first_step, best_loss, reproducible = restore_checkpoint_state(
        checkpoint,
        resumed_model,
        resumed_optimizer,
        resumed_train,
        resumed_validation,
    )
    resumed_loss, resumed_inputs = take_step(resumed_model, resumed_optimizer, resumed_train)
    resumed_random = (random.random(), np.random.random(), torch.rand(()))
    resumed_validation_batch = resumed_validation.get_batch(1, 8, torch.device("cpu"))[0]

    assert (first_step, best_loss, reproducible) == (1, 1.25, True)
    assert torch.equal(continuous_inputs, resumed_inputs)
    assert torch.equal(continuous_loss, resumed_loss)
    for name, parameter in resumed_model.named_parameters():
        assert torch.equal(continuous_parameters[name], parameter)
    assert continuous_random[0] == resumed_random[0]
    assert continuous_random[1] == resumed_random[1]
    assert torch.equal(continuous_random[2], resumed_random[2])
    assert torch.equal(continuous_validation, resumed_validation_batch)


def test_config_round_trip_serializes_every_field(tmp_path) -> None:
    import json

    config = ModelConfig(
        vocab_size=97,
        max_seq_len=19,
        n_layers=2,
        d_model=24,
        n_heads=3,
        n_kv_heads=1,
        d_ff=71,
        rope_theta=12_345.0,
        rms_norm_eps=2e-5,
        dropout=0.125,
        bias=True,
        tie_embeddings=True,
    )
    path = tmp_path / "model-config.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")

    restored = ModelConfig.from_json(path)

    assert restored == config
    assert set(restored.to_dict()) == set(ModelConfig.__dataclass_fields__)


def test_fixed_batch_logits_and_loss_are_deterministic_with_declared_float32_tolerance() -> None:
    tolerance = {"rtol": 1e-5, "atol": 1e-6}
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=8,
        n_layers=2,
        d_model=16,
        n_heads=2,
        n_kv_heads=1,
        d_ff=32,
        dropout=0.0,
    )
    torch.manual_seed(1234)
    model = TinyBenchLM(config).eval()
    input_ids = torch.tensor([[1, 5, 2, 9, 3], [4, 8, 7, 6, 0]])
    targets = torch.tensor([[5, 2, 9, 3, 1], [8, 7, 6, 0, 4]])

    first_logits, first_loss = model(input_ids, targets)
    second_logits, second_loss = model(input_ids, targets)

    assert first_loss is not None and second_loss is not None
    torch.testing.assert_close(first_logits, second_logits, **tolerance)
    torch.testing.assert_close(first_loss, second_loss, **tolerance)


def test_future_token_perturbation_does_not_change_earlier_logits() -> None:
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=8,
        n_layers=2,
        d_model=16,
        n_heads=2,
        n_kv_heads=1,
        d_ff=32,
        dropout=0.0,
    )
    torch.manual_seed(19)
    model = TinyBenchLM(config).eval()
    original = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed = original.clone()
    changed[:, 3:] = torch.tensor([[7, 8, 9]])

    original_logits, _ = model(original)
    changed_logits, _ = model(changed)

    torch.testing.assert_close(
        original_logits[:, :3],
        changed_logits[:, :3],
        rtol=0.0,
        atol=0.0,
    )


def test_unpadded_and_mixed_padding_loss_follow_explicit_ignore_policy() -> None:
    import torch.nn.functional as F

    from tinybench_lm import LOSS_IGNORE_INDEX

    config = ModelConfig(
        vocab_size=24,
        max_seq_len=8,
        n_layers=1,
        d_model=16,
        n_heads=2,
        n_kv_heads=1,
        d_ff=32,
    )
    torch.manual_seed(7)
    model = TinyBenchLM(config)
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    unpadded_targets = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]])

    logits, unpadded_loss = model(input_ids, unpadded_targets)
    expected_unpadded = F.cross_entropy(
        logits.float().reshape(-1, config.vocab_size),
        unpadded_targets.reshape(-1),
    )
    assert unpadded_loss is not None
    torch.testing.assert_close(unpadded_loss, expected_unpadded, rtol=0.0, atol=0.0)

    mixed_targets = unpadded_targets.clone()
    mixed_targets[0, 1] = LOSS_IGNORE_INDEX
    mixed_targets[1, 3] = LOSS_IGNORE_INDEX
    mixed_logits, mixed_loss = model(input_ids, mixed_targets)
    assert mixed_loss is not None and torch.isfinite(mixed_loss)
    mixed_logits.retain_grad()
    expected_mixed = F.cross_entropy(
        mixed_logits.float().reshape(-1, config.vocab_size),
        mixed_targets.reshape(-1),
        ignore_index=LOSS_IGNORE_INDEX,
    )
    torch.testing.assert_close(mixed_loss, expected_mixed, rtol=0.0, atol=0.0)

    mixed_loss.backward()
    assert mixed_logits.grad is not None
    ignored_positions = mixed_targets == LOSS_IGNORE_INDEX
    assert torch.count_nonzero(mixed_logits.grad[ignored_positions]) == 0


def test_all_padding_batch_has_finite_zero_loss_and_zero_gradient() -> None:
    from tinybench_lm import LOSS_IGNORE_INDEX

    config = ModelConfig(
        vocab_size=24,
        max_seq_len=8,
        n_layers=1,
        d_model=16,
        n_heads=2,
        n_kv_heads=1,
        d_ff=32,
    )
    torch.manual_seed(11)
    model = TinyBenchLM(config)
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    targets = torch.full_like(input_ids, LOSS_IGNORE_INDEX)

    _, loss = model(input_ids, targets)

    assert loss is not None and torch.isfinite(loss) and loss.item() == 0.0
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0


def test_final_config_projection_shapes_and_architecture_components() -> None:
    import torch.nn as nn

    from tinybench_lm.model import RMSNorm, SwiGLU

    config = ModelConfig.from_json(FINAL_CONFIG_PATH)
    model = TinyBenchLM(config)
    block = model.layers[0]

    assert len(model.layers) == 14
    assert block.attn.q_proj.weight.shape == (512, 512)
    assert block.attn.k_proj.weight.shape == (256, 512)
    assert block.attn.v_proj.weight.shape == (256, 512)
    assert block.attn.o_proj.weight.shape == (512, 512)
    assert block.ffn.gate_proj.weight.shape == (1_504, 512)
    assert block.ffn.up_proj.weight.shape == (1_504, 512)
    assert block.ffn.down_proj.weight.shape == (512, 1_504)
    assert block.attn.dropout == config.dropout == 0.0
    assert isinstance(block.attn_norm, RMSNorm)
    assert isinstance(block.ffn_norm, RMSNorm)
    assert isinstance(block.ffn, SwiGLU)
    assert isinstance(model.final_norm, RMSNorm)
    assert all(module.bias is None for module in model.modules() if isinstance(module, nn.Linear))


def test_gqa_expansion_repeats_each_kv_head_for_its_query_group() -> None:
    from tinybench_lm.model import CausalSelfAttention

    config = ModelConfig(
        vocab_size=16,
        max_seq_len=4,
        n_layers=1,
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        d_ff=32,
    )
    attention = CausalSelfAttention(config)
    key = torch.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4)
    value = key + 100

    expanded_key, expanded_value = attention._expand_kv_heads(key, value)

    assert expanded_key.shape == expanded_value.shape == (1, 4, 3, 4)
    assert torch.equal(expanded_key, key.repeat_interleave(2, dim=1))
    assert torch.equal(expanded_value, value.repeat_interleave(2, dim=1))
    assert torch.equal(expanded_key[:, 0], expanded_key[:, 1])
    assert torch.equal(expanded_key[:, 2], expanded_key[:, 3])


def test_transformer_block_is_pre_norm_and_ffn_is_swiglu() -> None:
    import torch.nn.functional as F

    from tinybench_lm.model import TransformerBlock

    config = ModelConfig(
        vocab_size=16,
        max_seq_len=4,
        n_layers=1,
        d_model=16,
        n_heads=2,
        n_kv_heads=1,
        d_ff=32,
        dropout=0.0,
    )
    torch.manual_seed(23)
    block = TransformerBlock(config).eval()
    hidden = torch.randn(2, 4, config.d_model)

    attention_output = block.attn(block.attn_norm(hidden))
    after_attention = hidden + attention_output
    ffn_input = block.ffn_norm(after_attention)
    expected = after_attention + block.ffn(ffn_input)
    actual = block(hidden)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    expected_swiglu = F.linear(
        F.silu(F.linear(ffn_input, block.ffn.gate_proj.weight))
        * F.linear(ffn_input, block.ffn.up_proj.weight),
        block.ffn.down_proj.weight,
    )
    torch.testing.assert_close(block.ffn(ffn_input), expected_swiglu, rtol=1e-5, atol=1e-6)


def test_rope_accepts_exact_context_and_rejects_longer_sequences() -> None:
    import pytest

    config = ModelConfig(
        vocab_size=16,
        max_seq_len=4,
        n_layers=1,
        d_model=16,
        n_heads=2,
        n_kv_heads=1,
        d_ff=32,
    )
    model = TinyBenchLM(config).eval()

    exact_logits, _ = model(torch.arange(4).reshape(1, 4))
    assert exact_logits.shape == (1, 4, config.vocab_size)
    with pytest.raises(ValueError, match="Sequence exceeds max_seq_len=4"):
        model(torch.arange(5).reshape(1, 5))


def test_gqa_expansion_matches_native_grouped_attention_and_is_identity_for_mha() -> None:
    """The Windows SDPA workaround must not change grouped-attention semantics."""
    import torch.nn.functional as F

    from tinybench_lm.model import CausalSelfAttention

    grouped_config = ModelConfig(
        vocab_size=16,
        max_seq_len=6,
        n_layers=1,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        dropout=0.0,
    )
    torch.manual_seed(31)
    attention = CausalSelfAttention(grouped_config).eval()
    hidden = torch.randn(2, 6, grouped_config.d_model)

    batch_size, seq_len, _ = hidden.shape
    head_dim = attention.head_dim
    q = (
        attention.q_proj(hidden)
        .view(batch_size, seq_len, grouped_config.n_heads, head_dim)
        .transpose(1, 2)
    )
    k = (
        attention.k_proj(hidden)
        .view(batch_size, seq_len, grouped_config.n_kv_heads, head_dim)
        .transpose(1, 2)
    )
    v = (
        attention.v_proj(hidden)
        .view(batch_size, seq_len, grouped_config.n_kv_heads, head_dim)
        .transpose(1, 2)
    )
    q, k = attention.rope(q, k)
    native = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    native_output = attention.o_proj(
        native.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    )

    torch.testing.assert_close(attention(hidden), native_output, rtol=1e-5, atol=1e-6)

    multi_head_config = ModelConfig(
        vocab_size=16,
        max_seq_len=6,
        n_layers=1,
        d_model=32,
        n_heads=4,
        n_kv_heads=4,
        d_ff=64,
        dropout=0.0,
    )
    multi_head_attention = CausalSelfAttention(multi_head_config)
    unexpanded_key, unexpanded_value = multi_head_attention._expand_kv_heads(k, v)
    assert unexpanded_key is k and unexpanded_value is v


def test_zero_dropout_makes_training_and_eval_forward_identical() -> None:
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=8,
        n_layers=2,
        d_model=16,
        n_heads=2,
        n_kv_heads=1,
        d_ff=32,
        dropout=0.0,
    )
    torch.manual_seed(41)
    model = TinyBenchLM(config)
    input_ids = torch.tensor([[3, 1, 4, 1, 5], [9, 2, 6, 5, 3]])
    targets = torch.tensor([[1, 4, 1, 5, 9], [2, 6, 5, 3, 5]])

    model.train()
    torch.manual_seed(0)
    train_logits, train_loss = model(input_ids, targets)
    model.eval()
    torch.manual_seed(0)
    eval_logits, eval_loss = model(input_ids, targets)

    assert train_loss is not None and eval_loss is not None
    assert all(block.attn.dropout == 0.0 for block in model.layers)
    torch.testing.assert_close(train_logits, eval_logits, rtol=0.0, atol=0.0)
    torch.testing.assert_close(train_loss, eval_loss, rtol=0.0, atol=0.0)


def test_padded_positions_contribute_zero_parameter_gradient() -> None:
    """Padding must not shift the loss or any parameter gradient of real tokens."""
    import torch.nn.functional as F

    from tinybench_lm import LOSS_IGNORE_INDEX

    tolerance = {"rtol": 1e-5, "atol": 1e-7}
    config = ModelConfig(
        vocab_size=24,
        max_seq_len=8,
        n_layers=2,
        d_model=16,
        n_heads=2,
        n_kv_heads=1,
        d_ff=32,
        dropout=0.0,
    )
    torch.manual_seed(53)
    model = TinyBenchLM(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    padded_targets = torch.tensor(
        [
            [2, 3, LOSS_IGNORE_INDEX, LOSS_IGNORE_INDEX],
            [6, LOSS_IGNORE_INDEX, 8, 9],
        ]
    )
    kept = padded_targets != LOSS_IGNORE_INDEX

    _, padded_loss = model(input_ids, padded_targets)
    assert padded_loss is not None
    model.zero_grad(set_to_none=True)
    padded_loss.backward()
    padded_gradients = {
        name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()
    }

    reference_logits, _ = model(input_ids)
    reference_loss = F.cross_entropy(
        reference_logits.float()[kept],
        padded_targets[kept],
    )
    model.zero_grad(set_to_none=True)
    reference_loss.backward()

    torch.testing.assert_close(padded_loss, reference_loss, **tolerance)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None
        torch.testing.assert_close(padded_gradients[name], parameter.grad, **tolerance)

    # Changing the ignored positions' input-independent target values is impossible,
    # so instead confirm that adding more padding leaves the kept-token loss intact.
    fully_kept_targets = padded_targets.clone()
    fully_kept_targets[0, 2] = 4
    _, wider_loss = model(input_ids, fully_kept_targets)
    assert wider_loss is not None
    assert not torch.isclose(wider_loss, padded_loss, rtol=0.0, atol=0.0)
    model.zero_grad(set_to_none=True)


def test_rope_cache_is_sized_to_max_seq_len_and_stays_out_of_the_state_dict() -> None:
    config = ModelConfig(
        vocab_size=16,
        max_seq_len=7,
        n_layers=1,
        d_model=16,
        n_heads=2,
        n_kv_heads=1,
        d_ff=32,
    )
    model = TinyBenchLM(config)
    rope = model.layers[0].attn.rope
    head_dim = config.d_model // config.n_heads

    assert rope.cos.shape == rope.sin.shape == (1, 1, config.max_seq_len, head_dim)
    assert not any("rope" in key for key in model.state_dict())


def test_odd_head_dimension_is_rejected_by_rope() -> None:
    import pytest

    with pytest.raises(ValueError, match="even attention head dimension"):
        TinyBenchLM(
            ModelConfig(
                vocab_size=16,
                max_seq_len=4,
                n_layers=1,
                d_model=6,
                n_heads=2,
                n_kv_heads=1,
                d_ff=12,
            )
        )
