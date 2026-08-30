from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, given, settings, strategies as st

from scripts.count_params import count_unique_trainable_parameters
from tinybench_lm import ModelConfig, TinyBenchLM
from tinybench_lm.checkpoint import restore_checkpoint_state, save_checkpoint
from tinybench_lm.config import COMPETITION_PARAMETER_CAP
from tinybench_lm.data import PackedTokenDataset
from tinybench_lm.provenance import (
    PROVENANCE_FORMAT_VERSION,
    RELEASE_FORBIDDEN_KEYS,
    RELEASE_FORMAT_VERSION,
    ProvenanceError,
    build_initial_model,
    declared_tolerance,
    export_release,
    export_release_from_checkpoint,
    fixed_batch,
    model_weight_sha256,
    optimizer_has_stepped,
    read_step_zero_provenance,
    record_step_zero_provenance,
    reject_resume_artifact_presented_as_fresh,
    reproduce_step_zero_weight_hash,
    resume_artifact_markers,
    state_dict_sha256,
    tied_output_shares_embedding_storage,
    verify_fresh_initialization_claim,
    verify_release_export,
    verify_step_zero_provenance,
    write_step_zero_provenance,
)


def tiny_config(**overrides: object) -> ModelConfig:
    settings_ = {
        "vocab_size": 32,
        "max_seq_len": 16,
        "n_layers": 2,
        "d_model": 16,
        "n_heads": 2,
        "n_kv_heads": 1,
        "d_ff": 32,
        "dropout": 0.0,
    }
    settings_.update(overrides)
    return ModelConfig(**settings_)  # type: ignore[arg-type]


def write_config(directory: Path, config: ModelConfig, name: str = "tiny.json") -> Path:
    path = directory / name
    path.write_text(json.dumps(config.to_dict(), sort_keys=True), encoding="utf-8")
    return path


def training_state(config: ModelConfig, token_path: Path, seed: int = 5):
    model = build_initial_model(config, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_data = PackedTokenDataset(token_path, seed=seed)
    validation_data = PackedTokenDataset(token_path, seed=seed + 1)
    return model, optimizer, train_data, validation_data


def write_tokens(path: Path, vocab_size: int, count: int = 1_024) -> Path:
    (np.arange(count, dtype=np.uint16) % vocab_size).tofile(path)
    return path


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_step_zero_provenance_records_seed_config_identifiers_and_weight_hash(tmp_path: Path) -> None:
    config = tiny_config()
    config_path = write_config(tmp_path, config)
    model = build_initial_model(config, seed=4242)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert not optimizer_has_stepped(optimizer)

    record = record_step_zero_provenance(
        model, config, seed=4242, config_path=config_path, optimizer=optimizer
    )

    assert record.seed == 4242
    assert record.initialization == "random"
    assert record.optimizer_steps_completed == 0
    assert record.provenance_format_version == PROVENANCE_FORMAT_VERSION
    assert record.model_config == config.to_dict()
    assert record.config_sha256 is not None and len(record.config_sha256) == 64
    assert record.weight_sha256 == model_weight_sha256(model)
    assert record.unique_parameter_count == count_unique_trainable_parameters(model)
    assert record.cap_headroom == COMPETITION_PARAMETER_CAP - record.unique_parameter_count
    assert record.tied_embedding_output_storage is True
    assert set(record.code_identifier) == {
        "git_commit",
        "production_source_tree_sha256",
        "torch_version",
    }
    assert len(record.code_identifier["production_source_tree_sha256"]) == 64

    path = write_step_zero_provenance(tmp_path / "step_zero_provenance.json", record)
    assert read_step_zero_provenance(path) == record

    report = verify_step_zero_provenance(record, model=model)
    assert report.ok, report.to_dict()
    assert report.result("provenance.random_initialization_reproducible").status == "PASS"
    assert report.result("provenance.supplied_model_weight_hash").status == "PASS"

    # The recorded hash is reproducible from configuration plus seed, and only that seed.
    assert reproduce_step_zero_weight_hash(config, 4242) == record.weight_sha256
    assert reproduce_step_zero_weight_hash(config, 4243) != record.weight_sha256


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2**
def test_step_zero_provenance_cannot_be_recorded_after_the_first_optimizer_step(tmp_path: Path) -> None:
    config = tiny_config()
    token_path = write_tokens(tmp_path / "tokens.bin", config.vocab_size)
    model, optimizer, train_data, _ = training_state(config, token_path)

    inputs, targets = train_data.get_batch(2, 8, torch.device("cpu"))
    _, loss = model(inputs, targets)
    assert loss is not None
    loss.backward()
    optimizer.step()

    assert optimizer_has_stepped(optimizer)
    with pytest.raises(ProvenanceError, match="before the first optimizer step"):
        record_step_zero_provenance(model, config, seed=5, optimizer=optimizer)


# **Validates: Requirements 2.2, 2.5, 3.3**
def test_existing_step_zero_evidence_is_never_silently_rewritten(tmp_path: Path) -> None:
    config = tiny_config()
    first = record_step_zero_provenance(build_initial_model(config, 1), config, seed=1)
    second = record_step_zero_provenance(build_initial_model(config, 2), config, seed=2)
    path = tmp_path / "step_zero_provenance.json"

    write_step_zero_provenance(path, first)
    # Re-recording the same claim is idempotent.
    assert write_step_zero_provenance(path, first) == path
    with pytest.raises(ProvenanceError, match="different step-zero claim"):
        write_step_zero_provenance(path, second)
    assert read_step_zero_provenance(path) == first
    assert write_step_zero_provenance(path, second, overwrite=True) == path
    assert read_step_zero_provenance(path) == second


# **Validates: Requirements 1.2, 2.2, 2.4, 2.5**
def test_resume_artifact_is_rejected_when_presented_as_a_fresh_initialization(tmp_path: Path) -> None:
    config = tiny_config()
    token_path = write_tokens(tmp_path / "tokens.bin", config.vocab_size)
    model, optimizer, train_data, validation_data = training_state(config, token_path)
    record = record_step_zero_provenance(model, config, seed=5, optimizer=optimizer)

    inputs, targets = train_data.get_batch(2, 8, torch.device("cpu"))
    _, loss = model(inputs, targets)
    assert loss is not None
    loss.backward()
    optimizer.step()
    checkpoint_path = tmp_path / "latest.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        config,
        argparse.Namespace(seed=5),
        train_data,
        validation_data,
        step=0,
        best_validation_loss=2.5,
    )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    markers = resume_artifact_markers(payload)
    assert set(markers) >= {"optimizer", "rng_state", "data_rng_state", "step"}

    report = verify_fresh_initialization_claim(payload, record=record, source=str(checkpoint_path))
    assert not report.ok
    assert report.result("fresh_initialization.no_resume_state").status == "FAIL"
    assert report.result("fresh_initialization.weight_hash").status == "FAIL"
    with pytest.raises(ProvenanceError, match="not a fresh initialization"):
        reject_resume_artifact_presented_as_fresh(
            payload, record=record, source=str(checkpoint_path)
        )

    # A genuine step-zero artifact with the same record is accepted.
    fresh_model = build_initial_model(config, 5)
    fresh_payload = {"model": fresh_model.state_dict(), "model_config": config.to_dict()}
    assert resume_artifact_markers(fresh_payload) == ()
    fresh_report = verify_fresh_initialization_claim(fresh_payload, record=record)
    assert fresh_report.ok, fresh_report.to_dict()
    reject_resume_artifact_presented_as_fresh(fresh_payload, record=record)

    # A fresh-initialization claim without step-zero evidence cannot pass.
    unevidenced = verify_fresh_initialization_claim(fresh_payload)
    assert not unevidenced.ok
    assert unevidenced.result("fresh_initialization.step_zero_record").status == "FAIL"


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_release_export_reloads_without_optimizer_state_and_matches_fixed_batch(tmp_path: Path) -> None:
    config = tiny_config()
    config_path = write_config(tmp_path, config)
    model = build_initial_model(config, seed=77)
    record = record_step_zero_provenance(model, config, seed=77, config_path=config_path)
    release_path = tmp_path / "release.pt"

    payload = export_release(release_path, model, provenance=record, notes="fixture-scale export")

    assert payload["release_format_version"] == RELEASE_FORMAT_VERSION
    assert not [key for key in RELEASE_FORBIDDEN_KEYS if key in payload]
    assert payload["unique_parameter_count"] == count_unique_trainable_parameters(model)
    assert payload["tied_embedding_output_storage"] is True
    assert payload["weight_sha256"] == model_weight_sha256(model)
    assert payload["fixed_batch"]["dtype"] == "torch.float32"
    assert (payload["fixed_batch"]["rtol"], payload["fixed_batch"]["atol"]) == declared_tolerance(
        torch.float32
    )
    inputs, targets = fixed_batch(config)
    assert torch.equal(payload["fixed_batch"]["input_ids"], inputs)
    assert torch.equal(payload["fixed_batch"]["targets"], targets)

    stored = torch.load(release_path, map_location="cpu", weights_only=False)
    assert not [key for key in RELEASE_FORBIDDEN_KEYS if key in stored]
    assert set(stored["model"]) == set(model.state_dict())

    report = verify_release_export(
        release_path, expected_parameter_count=payload["unique_parameter_count"]
    )
    assert report.ok, report.to_dict()
    for check_id in (
        "release.no_optimizer_state",
        "release.reload",
        "release.unique_parameter_count",
        "release.parameter_cap",
        "release.expected_parameter_count",
        "release.tied_embedding_output_storage",
        "release.weight_sha256",
        "release.fixed_batch_logits",
        "release.fixed_batch_loss",
        "release.provenance_config_agreement",
        "release.provenance_parameter_count",
        "provenance.random_initialization_reproducible",
    ):
        assert report.result(check_id).status == "PASS"

    reloaded = TinyBenchLM(ModelConfig(**stored["model_config"]))
    reloaded.load_state_dict(stored["model"])
    assert tied_output_shares_embedding_storage(reloaded)


# **Validates: Requirements 1.1, 1.2, 2.1, 2.4, 2.5**
def test_release_verification_fails_closed_on_tampered_or_unevidenced_exports(tmp_path: Path) -> None:
    config = tiny_config()
    model = build_initial_model(config, seed=91)
    record = record_step_zero_provenance(model, config, seed=91)
    good_path = tmp_path / "release.pt"
    export_release(good_path, model, provenance=record)

    # Optimizer state smuggled back into a release export.
    with_optimizer = torch.load(good_path, map_location="cpu", weights_only=False)
    with_optimizer["optimizer"] = {"state": {0: {"step": 1}}}
    optimizer_path = tmp_path / "release-with-optimizer.pt"
    torch.save(with_optimizer, optimizer_path)
    optimizer_report = verify_release_export(optimizer_path)
    assert not optimizer_report.ok
    assert optimizer_report.result("release.no_optimizer_state").status == "FAIL"

    # Weights that no longer reproduce the exported fixed-batch fingerprint.
    tampered = torch.load(good_path, map_location="cpu", weights_only=False)
    tampered["model"]["token_embedding.weight"] = (
        tampered["model"]["token_embedding.weight"] + 0.5
    )
    tampered_path = tmp_path / "release-tampered.pt"
    torch.save(tampered, tampered_path)
    tampered_report = verify_release_export(tampered_path)
    assert not tampered_report.ok
    assert tampered_report.result("release.weight_sha256").status == "FAIL"
    assert tampered_report.result("release.fixed_batch_logits").status == "FAIL"
    assert tampered_report.result("release.fixed_batch_loss").status == "FAIL"

    # An export with no step-zero evidence cannot claim random initialization.
    unevidenced_path = tmp_path / "release-unevidenced.pt"
    export_release(unevidenced_path, model)
    unevidenced_report = verify_release_export(unevidenced_path)
    assert not unevidenced_report.ok
    assert unevidenced_report.result("release.step_zero_provenance").status == "FAIL"

    # An externally expected count that disagrees with the artifact fails.
    mismatch_report = verify_release_export(good_path, expected_parameter_count=1)
    assert not mismatch_report.ok
    assert mismatch_report.result("release.expected_parameter_count").status == "FAIL"


# **Validates: Requirements 2.4, 2.5, 3.3**
def test_release_export_from_local_checkpoint_preserves_resume_loading(tmp_path: Path) -> None:
    config = tiny_config()
    token_path = write_tokens(tmp_path / "tokens.bin", config.vocab_size)
    model, optimizer, train_data, validation_data = training_state(config, token_path, seed=13)
    record = record_step_zero_provenance(model, config, seed=13, optimizer=optimizer)
    provenance_path = write_step_zero_provenance(tmp_path / "step_zero_provenance.json", record)

    inputs, targets = train_data.get_batch(2, 8, torch.device("cpu"))
    _, loss = model(inputs, targets)
    assert loss is not None
    loss.backward()
    optimizer.step()
    checkpoint_path = tmp_path / "best.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        config,
        argparse.Namespace(seed=13),
        train_data,
        validation_data,
        step=0,
        best_validation_loss=1.5,
    )

    release_path = tmp_path / "release.pt"
    payload = export_release_from_checkpoint(
        checkpoint_path, release_path, provenance_path=provenance_path
    )
    assert payload["weight_sha256"] == state_dict_sha256(model.state_dict())
    report = verify_release_export(release_path)
    assert report.result("release.fixed_batch_logits").status == "PASS"
    assert report.result("release.fixed_batch_loss").status == "PASS"
    assert report.result("release.no_optimizer_state").status == "PASS"
    # Trained weights are no longer step zero, which the provenance check must say plainly.
    assert report.result("provenance.random_initialization_reproducible").status == "PASS"

    # Preservation: the local checkpoint still resumes exactly as before.
    resumed_model, resumed_optimizer, resumed_train, resumed_validation = training_state(
        config, token_path, seed=13
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    first_step, best_loss, reproducible = restore_checkpoint_state(
        checkpoint, resumed_model, resumed_optimizer, resumed_train, resumed_validation
    )
    assert (first_step, best_loss, reproducible) == (1, 1.5, True)
    assert state_dict_sha256(resumed_model.state_dict()) == state_dict_sha256(model.state_dict())


@st.composite
def provenance_cases(draw: st.DrawFn) -> tuple[ModelConfig, int]:
    d_model, n_heads, n_kv_heads = draw(
        st.sampled_from(((8, 1, 1), (16, 2, 1), (16, 4, 2), (32, 4, 2)))
    )
    config = ModelConfig(
        vocab_size=draw(st.integers(min_value=8, max_value=48)),
        max_seq_len=draw(st.integers(min_value=2, max_value=12)),
        n_layers=draw(st.integers(min_value=1, max_value=2)),
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_ff=draw(st.integers(min_value=d_model, max_value=2 * d_model)),
        dropout=0.0,
    )
    return config, draw(st.integers(min_value=0, max_value=2**31 - 1))


# **Validates: Requirements 2.1, 2.4, 2.5**
@given(case=provenance_cases())
@settings(
    max_examples=6,
    deadline=None,
    derandomize=True,
    suppress_health_check=(HealthCheck.too_slow,),
)
def test_step_zero_and_release_evidence_hold_for_any_valid_tiny_initialization(
    case: tuple[ModelConfig, int],
) -> None:
    config, seed = case
    model = build_initial_model(config, seed, isolate_global_rng=True)
    record = record_step_zero_provenance(model, config, seed=seed)

    assert record.weight_sha256 == reproduce_step_zero_weight_hash(config, seed)
    assert reproduce_step_zero_weight_hash(config, seed + 1) != record.weight_sha256
    assert verify_step_zero_provenance(record, model=model).ok

    with TemporaryDirectory() as directory:
        release_path = Path(directory) / "release.pt"
        payload = export_release(release_path, model, provenance=record)
        assert payload["unique_parameter_count"] == count_unique_trainable_parameters(model)
        report = verify_release_export(
            release_path, expected_parameter_count=record.unique_parameter_count
        )
        assert report.ok, report.to_dict()


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_training_entry_point_records_step_zero_evidence_before_the_first_optimizer_step() -> None:
    """The record must be wired into train.py ahead of the update loop, not just available."""
    import ast

    repository_root = Path(__file__).resolve().parents[1]
    source = (repository_root / "train.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="train.py")

    record_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "record_step_zero_provenance"
    ]
    assert len(record_calls) == 1
    assert {keyword.arg for keyword in record_calls[0].keywords} >= {
        "seed",
        "config_path",
        "optimizer",
    }

    write_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "write_step_zero_provenance"
    ]
    assert len(write_calls) == 1

    training_loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "step"
    ]
    assert training_loops
    first_update_line = min(loop.lineno for loop in training_loops)
    assert record_calls[0].lineno < first_update_line
    assert write_calls[0].lineno < first_update_line
