from __future__ import annotations

import ast
import random
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from hypothesis import HealthCheck, given, settings, strategies as st

from tinybench_lm import ModelConfig, TinyBenchLM
from tinybench_lm.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    capture_rng_state,
    restore_rng_state,
)
from tinybench_lm.data import PackedTokenDataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# Observation-first baseline recorded before alignment implementation:
# - valid tiny configurations produce finite logits and loss;
# - causal masking isolates earlier logits from future-token changes;
# - output projection reuses the input embedding parameter;
# - generation appends exactly the requested number of tokens;
# - format-v2 resume restores global RNG and sampler state exactly;
# - untied embeddings are rejected;
# - eligible production sources contain no model-weight from_pretrained() call;
# - existing entry points and factual pilot reports are present.


@st.composite
def tiny_model_cases(draw: st.DrawFn) -> tuple[ModelConfig, torch.Tensor, torch.Tensor, int]:
    d_model = draw(st.sampled_from((8, 16, 24, 32)))
    valid_head_counts = [
        heads
        for heads in (1, 2, 4, 8)
        if d_model % heads == 0 and (d_model // heads) % 2 == 0
    ]
    n_heads = draw(st.sampled_from(valid_head_counts))
    n_kv_heads = draw(st.sampled_from([heads for heads in valid_head_counts if n_heads % heads == 0]))
    max_seq_len = draw(st.integers(min_value=3, max_value=12))
    sequence_length = draw(st.integers(min_value=2, max_value=max_seq_len))
    batch_size = draw(st.integers(min_value=1, max_value=3))
    vocab_size = draw(st.integers(min_value=8, max_value=64))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))

    config = ModelConfig(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        n_layers=draw(st.integers(min_value=1, max_value=2)),
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_ff=draw(st.integers(min_value=d_model, max_value=4 * d_model)),
        dropout=0.0,
    )
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(
        0,
        vocab_size,
        (batch_size, sequence_length),
        generator=generator,
    )
    targets = torch.randint(
        0,
        vocab_size,
        (batch_size, sequence_length),
        generator=generator,
    )
    first_changed_position = draw(st.integers(min_value=1, max_value=sequence_length - 1))
    return config, input_ids, targets, first_changed_position


# **Validates: Requirements 3.1, 3.3**
@given(case=tiny_model_cases())
@settings(
    max_examples=8,
    deadline=None,
    derandomize=True,
    suppress_health_check=(HealthCheck.too_slow,),
)
def test_valid_tiny_models_preserve_forward_causality_tying_and_generation(
    case: tuple[ModelConfig, torch.Tensor, torch.Tensor, int],
) -> None:
    config, input_ids, targets, first_changed_position = case
    torch.manual_seed(0)
    model = TinyBenchLM(config).eval()

    logits, loss = model(input_ids, targets)
    assert logits.shape == (*input_ids.shape, config.vocab_size)
    assert torch.isfinite(logits).all()
    assert loss is not None and torch.isfinite(loss)

    changed = input_ids.clone()
    changed[:, first_changed_position:] = (
        changed[:, first_changed_position:] + 1
    ) % config.vocab_size
    changed_logits, _ = model(changed)
    torch.testing.assert_close(
        logits[:, :first_changed_position],
        changed_logits[:, :first_changed_position],
        rtol=0.0,
        atol=0.0,
    )

    hidden = model.token_embedding(input_ids)
    for layer in model.layers:
        hidden = layer(hidden)
    hidden = model.final_norm(hidden)
    tied_projection = F.linear(hidden, model.token_embedding.weight)
    torch.testing.assert_close(logits, tied_projection, rtol=0.0, atol=0.0)
    assert "token_embedding.weight" in dict(model.named_parameters())
    assert not any(name.startswith("lm_head") for name, _ in model.named_parameters())

    prompt = input_ids[:, :1]
    generated = model.generate(prompt, max_new_tokens=2, temperature=1.0, top_k=4)
    assert generated.shape == (input_ids.size(0), prompt.size(1) + 2)
    assert torch.equal(generated[:, : prompt.size(1)], prompt)
    assert torch.all((0 <= generated) & (generated < config.vocab_size))


# **Validates: Requirements 3.1, 3.3**
@given(
    global_seed=st.integers(min_value=0, max_value=2**31 - 1),
    sampler_seed=st.integers(min_value=0, max_value=2**31 - 1),
    batch_size=st.integers(min_value=1, max_value=3),
    sequence_length=st.integers(min_value=2, max_value=12),
)
@settings(max_examples=8, deadline=None, derandomize=True)
def test_format_v2_rng_and_sampler_resume_is_exact(
    global_seed: int,
    sampler_seed: int,
    batch_size: int,
    sequence_length: int,
) -> None:
    assert CHECKPOINT_FORMAT_VERSION == 2
    with TemporaryDirectory() as temporary_directory:
        token_path = Path(temporary_directory) / (
            f"tokens-{global_seed}-{sampler_seed}-{batch_size}-{sequence_length}.bin"
        )
        (np.arange(512, dtype=np.uint16) % 61).tofile(token_path)
        dataset = PackedTokenDataset(token_path, seed=sampler_seed)

        try:
            # Advance once so the preserved state is not merely the initial seed state.
            dataset.get_batch(batch_size, sequence_length, torch.device("cpu"))
            random.seed(global_seed)
            np.random.seed(global_seed)
            torch.manual_seed(global_seed)
            rng_state = capture_rng_state()
            sampler_state = dataset.state_dict()

            expected_batch = dataset.get_batch(batch_size, sequence_length, torch.device("cpu"))
            expected_random = (random.random(), np.random.random(), torch.rand(3))

            random.random()
            np.random.random()
            torch.rand(3)
            dataset.get_batch(batch_size, sequence_length, torch.device("cpu"))

            restore_rng_state(rng_state)
            dataset.load_state_dict(sampler_state)
            resumed_batch = dataset.get_batch(batch_size, sequence_length, torch.device("cpu"))
            resumed_random = (random.random(), np.random.random(), torch.rand(3))

            assert torch.equal(expected_batch[0], resumed_batch[0])
            assert torch.equal(expected_batch[1], resumed_batch[1])
            assert expected_random[0] == resumed_random[0]
            assert expected_random[1] == resumed_random[1]
            assert torch.equal(expected_random[2], resumed_random[2])
        finally:
            dataset.tokens._mmap.close()


# **Validates: Requirements 3.1, 3.3**
def test_untied_embeddings_remain_rejected() -> None:
    with pytest.raises(ValueError, match="Untied embeddings"):
        ModelConfig(
            vocab_size=32,
            max_seq_len=8,
            n_layers=1,
            d_model=16,
            n_heads=2,
            n_kv_heads=1,
            d_ff=32,
            tie_embeddings=False,
        )


def _production_python_paths() -> list[Path]:
    paths = [REPOSITORY_ROOT / name for name in ("train.py", "generate.py", "evaluate.py")]
    paths.extend(sorted((REPOSITORY_ROOT / "src" / "tinybench_lm").glob("*.py")))
    paths.extend(sorted((REPOSITORY_ROOT / "scripts").glob("*.py")))
    return paths


# **Validates: Requirements 3.2, 3.3**
def test_eligible_production_sources_do_not_load_pretrained_model_weights() -> None:
    violations: list[str] = []
    for path in _production_python_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "from_pretrained"
            ):
                violations.append(f"{path.relative_to(REPOSITORY_ROOT)}:{node.lineno}")
    assert violations == []


# **Validates: Requirements 3.1, 3.2, 3.4**
def test_entry_points_and_factual_pilot_reports_remain_present() -> None:
    entry_points = ("train.py", "generate.py", "evaluate.py")
    for relative_path in entry_points:
        path = REPOSITORY_ROOT / relative_path
        assert path.is_file()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assert any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main"
            for node in tree.body
        )

    report_markers = {
        "docs/PILOT_REPORT.md": (
            "# TinyBench-LM local pilot report",
            "## Measured hardware performance",
            "This pilot validates engineering, not model capability.",
        ),
        "docs/PHASE_1_BASELINE.md": (
            "# Phase 1 baseline and rules lock",
            "format-version-2 checkpoints preserve those states",
            "No full benchmark or long training run was started.",
        ),
    }
    for relative_path, markers in report_markers.items():
        path = REPOSITORY_ROOT / relative_path
        assert path.is_file()
        content = path.read_text(encoding="utf-8")
        assert all(marker in content for marker in markers)
