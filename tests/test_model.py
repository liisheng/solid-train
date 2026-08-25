import argparse
import random

import numpy as np
import torch

from tinybench_lm import ModelConfig, TinyBenchLM
from tinybench_lm.checkpoint import restore_checkpoint_state, save_checkpoint
from tinybench_lm.data import PackedTokenDataset


def test_baseline_stays_under_parameter_cap() -> None:
    config = ModelConfig.from_json("configs/baseline_49m.json")
    model = TinyBenchLM(config)
    assert model.count_parameters() == 49_295_872
    assert model.count_parameters() <= 50_000_000


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
