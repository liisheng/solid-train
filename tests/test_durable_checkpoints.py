"""Task 3.13: durable, verified, schedule-aware checkpoints.

Every test here runs on tiny CPU fixtures. Nothing starts training, measures checkpoint wall
time, or verifies an off-machine copy: those stay operator actions reported as NOT_RUN.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st

from tinybench_lm import ModelConfig
from tinybench_lm.checkpoint import CHECKPOINT_FORMAT_VERSION, save_checkpoint
from tinybench_lm.checkpointing import (
    CHECKPOINT_CHECKSUM_MISMATCH,
    CHECKPOINT_INCOMPLETE_STATE,
    CHECKPOINT_MANIFEST_MISSING,
    CHECKPOINT_NOT_AT_ACCUMULATION_BOUNDARY,
    CHECKPOINT_RETENTION_UNVERIFIED_RETAINED,
    CHECKPOINT_ROLES,
    DELETE,
    DURABLE_CHECKPOINT_FORMAT_VERSION,
    FROZEN_CHECKPOINT_PROTOCOL_SHA256,
    KEEP,
    PILOT_SCHEDULE_HASH_SENTINEL,
    PROTECTED_ROLES,
    ROLE_BRANCH_PARENT,
    ROLE_FALLBACK,
    ROLE_RECOVERY,
    ROLE_SELECTED_ENDPOINT,
    TEMPORARY_SUFFIX,
    BestValidationState,
    CheckpointContractError,
    CheckpointCounters,
    CheckpointEntry,
    CheckpointIntegrityError,
    assert_accumulation_boundary,
    assert_resume_matches_frozen_artifacts,
    assert_retention_plan_safe,
    build_checkpoint_payload,
    checkpoint_protocol_digest,
    frozen_config_hashes,
    inventory_from_directory,
    load_checkpoint_protocol,
    load_verified_checkpoint,
    manifest_path_for,
    payload_violations,
    plan_retention,
    read_manifest,
    readiness_results,
    required_payload_keys,
    restore_durable_state,
    retention_plan_violations,
    save_durable_checkpoint,
    verify_checkpoint,
)
from tinybench_lm.data import PackedTokenDataset
from tinybench_lm.provenance import build_initial_model, state_dict_sha256
from tinybench_lm.schedule import ScheduleCursor, ScheduleResumeError
from tinybench_lm.training_recipe import (
    BatchPlan,
    TrainingIntegrityError,
    model_config_hash,
)

SEED = 7
PLAN = BatchPlan(micro_batch_size=2, sequence_length=8, gradient_accumulation=2)


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        max_seq_len=16,
        n_layers=2,
        d_model=16,
        n_heads=2,
        n_kv_heads=1,
        d_ff=32,
        dropout=0.0,
    )


def write_tokens(path: Path, vocab_size: int, count: int = 2_048) -> Path:
    (np.arange(count, dtype=np.uint16) % vocab_size).tofile(path)
    return path


def training_state(config: ModelConfig, token_path: Path, seed: int = SEED):
    """A fresh tiny CPU training state: seeded model, AdamW, and two pilot samplers."""
    model = build_initial_model(config, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_data = PackedTokenDataset(token_path, seed=seed)
    validation_data = PackedTokenDataset(token_path, seed=seed + 1)
    return model, optimizer, train_data, validation_data


def take_update(model, optimizer, train_data, plan: BatchPlan = PLAN) -> float:
    """One complete optimizer update over ``plan.gradient_accumulation`` microbatches."""
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for _ in range(plan.gradient_accumulation):
        inputs, targets = train_data.get_batch(
            plan.micro_batch_size, plan.sequence_length, torch.device("cpu")
        )
        _, loss = model(inputs, targets)
        assert loss is not None
        (loss / plan.gradient_accumulation).backward()
        total += float(loss.detach()) / plan.gradient_accumulation
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return total


def artifact_hashes(config: ModelConfig) -> dict[str, str]:
    return frozen_config_hashes(model_config_hash=model_config_hash(config.to_dict()))


def make_payload(
    config: ModelConfig,
    model,
    optimizer,
    train_data,
    validation_data,
    *,
    update_index: int,
    run_id: str = "run-fixture0000000",
    scaler=None,
    best_validation: BestValidationState | None = None,
    microbatches_completed_in_update: int = 0,
    schedule_content_hash: str = PILOT_SCHEDULE_HASH_SENTINEL,
) -> dict[str, object]:
    return build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        config=config,
        args=argparse.Namespace(seed=SEED, steps=8),
        train_data=train_data,
        validation_data=validation_data,
        counters=CheckpointCounters.at_update(
            update_index, PLAN, microbatches_completed_in_update=microbatches_completed_in_update
        ),
        run_id=run_id,
        frozen_config_hashes=artifact_hashes(config),
        schedule_content_hash=schedule_content_hash,
        best_validation=best_validation or BestValidationState.unevaluated(),
    )


# --------------------------------------------------------------------------------------
# The frozen contract
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_frozen_checkpoint_protocol_is_pinned_and_declares_the_full_contract() -> None:
    protocol = load_checkpoint_protocol()
    assert protocol["_digest"] == FROZEN_CHECKPOINT_PROTOCOL_SHA256["checkpoint_v1.yaml"]
    assert protocol["frozen"] is True
    assert protocol["boundary"]["save_only_at_accumulation_boundaries"] is True

    keys = required_payload_keys(protocol)
    for required in (
        "model",
        "optimizer",
        "scaler",
        "rng_state",
        "data_rng_state",
        "schedule_content_hash",
        "schedule_cursor",
        "counters",
        "run_id",
        "frozen_config_hashes",
        "best_validation_state",
    ):
        assert required in keys

    assert tuple(protocol["durability"]["write_order"]) == (
        "write_temporary_file",
        "flush",
        "fsync",
        "checksum",
        "independent_load_test",
        "rename",
    )
    assert protocol["retention"]["keep_latest_recovery_states"] == 3
    assert {str(role["role"]) for role in protocol["retention"]["roles"]} == set(CHECKPOINT_ROLES)
    assert protocol["retention"]["never_delete_unverified_local_copy"] is True
    assert protocol["retention"]["never_delete_unverified_remote_copy"] is True

    # A durable payload stays a superset of the proven format-v2 envelope.
    assert protocol["payload"]["base_format_version"] == CHECKPOINT_FORMAT_VERSION
    assert protocol["payload"]["durable_format_version"] == DURABLE_CHECKPOINT_FORMAT_VERSION


# **Validates: Requirements 2.4, 2.5**
def test_unmeasured_checkpoint_evidence_is_reported_as_not_run() -> None:
    statuses = {result.check_id: result.status for result in readiness_results()}
    assert statuses["checkpoint.readiness.verified_remote_copy"] == "NOT_RUN"
    assert statuses["checkpoint.readiness.measured_checkpoint_write_seconds"] == "NOT_RUN"
    assert statuses["checkpoint.readiness.measured_checkpoint_size_bytes"] == "NOT_RUN"
    assert "PASS" not in set(statuses.values())


# --------------------------------------------------------------------------------------
# Complete state
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_durable_payload_carries_model_optimizer_scaler_rng_schedule_counters_and_lineage(
    tmp_path: Path,
) -> None:
    config = tiny_config()
    token_path = write_tokens(tmp_path / "tokens.bin", config.vocab_size)
    model, optimizer, train_data, validation_data = training_state(config, token_path)
    take_update(model, optimizer, train_data)
    scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=4096.0)

    payload = make_payload(
        config,
        model,
        optimizer,
        train_data,
        validation_data,
        update_index=0,
        scaler=scaler,
        best_validation=BestValidationState(loss=2.5, update_index=0, relative_path="best.pt"),
    )

    assert payload_violations(payload) == ()
    assert payload["checkpoint_format_version"] == CHECKPOINT_FORMAT_VERSION
    assert payload["durable_checkpoint_format_version"] == DURABLE_CHECKPOINT_FORMAT_VERSION
    assert payload["protocol_digest"] == checkpoint_protocol_digest()
    assert set(payload["rng_state"]) >= {"python", "numpy", "torch_cpu"}
    assert set(payload["data_rng_state"]) == {"train", "validation"}
    assert payload["grad_scaler_enabled"] is True
    assert payload["scaler"]["scale"] == 4096.0
    assert payload["counters"] == {
        "update_index": 0,
        "updates_completed": 1,
        "consumed_loss_tokens": 32,
        "loss_tokens_per_update": 32,
        "gradient_accumulation": 2,
        "microbatches_completed_in_update": 0,
    }
    assert payload["run_id"] == "run-fixture0000000"
    assert set(payload["frozen_config_hashes"]) >= {
        "recipe_digest",
        "model_config_hash",
        "checkpoint_protocol_digest",
    }
    assert payload["weight_sha256"] == state_dict_sha256(model.state_dict())
    assert payload["best_validation_state"]["loss"] == 2.5
    # The pilot flat-stream sampler has no integer cursor, and that absence is labeled.
    assert payload["schedule_cursor"] is None
    assert payload["schedule_content_hash"] == PILOT_SCHEDULE_HASH_SENTINEL


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_missing_state_and_impossible_counters_fail_closed(tmp_path: Path) -> None:
    config = tiny_config()
    token_path = write_tokens(tmp_path / "tokens.bin", config.vocab_size)
    model, optimizer, train_data, validation_data = training_state(config, token_path)
    payload = make_payload(config, model, optimizer, train_data, validation_data, update_index=0)

    for dropped in ("scaler", "counters", "rng_state", "run_id", "frozen_config_hashes"):
        incomplete = {key: value for key, value in payload.items() if key != dropped}
        problems = payload_violations(incomplete)
        assert any(CHECKPOINT_INCOMPLETE_STATE in problem for problem in problems), dropped

    tampered = dict(payload)
    tampered["counters"] = {**payload["counters"], "consumed_loss_tokens": 99}
    assert any("COUNTER_IMPOSSIBLE" in problem for problem in payload_violations(tampered))

    enabled_without_state = dict(payload)
    enabled_without_state["grad_scaler_enabled"] = True
    enabled_without_state["scaler"] = None
    assert any(CHECKPOINT_INCOMPLETE_STATE in problem for problem in payload_violations(enabled_without_state))


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_mid_accumulation_save_is_refused(tmp_path: Path) -> None:
    config = tiny_config()
    token_path = write_tokens(tmp_path / "tokens.bin", config.vocab_size)
    model, optimizer, train_data, validation_data = training_state(config, token_path)

    with pytest.raises(CheckpointContractError, match=CHECKPOINT_NOT_AT_ACCUMULATION_BOUNDARY):
        make_payload(
            config,
            model,
            optimizer,
            train_data,
            validation_data,
            update_index=0,
            microbatches_completed_in_update=1,
        )
    with pytest.raises(CheckpointContractError, match=CHECKPOINT_NOT_AT_ACCUMULATION_BOUNDARY):
        assert_accumulation_boundary(1, PLAN.gradient_accumulation)
    assert_accumulation_boundary(0, PLAN.gradient_accumulation) is None


# --------------------------------------------------------------------------------------
# Durable write and verification
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_save_is_atomic_checksummed_load_tested_and_verifiable(tmp_path: Path) -> None:
    config = tiny_config()
    token_path = write_tokens(tmp_path / "tokens.bin", config.vocab_size)
    model, optimizer, train_data, validation_data = training_state(config, token_path)
    take_update(model, optimizer, train_data)
    payload = make_payload(config, model, optimizer, train_data, validation_data, update_index=0)

    path = tmp_path / "run" / "latest.pt"
    manifest = save_durable_checkpoint(path, payload)

    assert path.is_file()
    assert manifest_path_for(path).is_file()
    # The temporary file is renamed, never left behind.
    assert list(path.parent.glob(f"*{TEMPORARY_SUFFIX}")) == []
    assert manifest.bytes == path.stat().st_size
    assert manifest.update_index == 0
    assert manifest.consumed_loss_tokens == 32
    assert manifest.weight_sha256 == payload["weight_sha256"]
    assert manifest.remote_copy_verified.startswith("NOT_RUN")
    assert read_manifest(path) == manifest

    report = verify_checkpoint(
        path,
        expected_run_id="run-fixture0000000",
        expected_frozen_config_hashes=artifact_hashes(config),
        expected_schedule_content_hash=PILOT_SCHEDULE_HASH_SENTINEL,
    )
    assert report.ok, report.to_dict()
    assert report.result("checkpoint.checksum").status == "PASS"
    assert report.result("checkpoint.payload_complete").status == "PASS"
    assert report.result("checkpoint.accumulation_boundary").status == "PASS"
    assert report.result("checkpoint.manifest_agrees_with_payload").status == "PASS"
    # Off-machine verification is an operator action and is never reported as a pass.
    assert report.result("checkpoint.remote_copy_verified").status == "NOT_RUN"


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_corruption_and_a_missing_manifest_fail_closed_before_resume(tmp_path: Path) -> None:
    config = tiny_config()
    token_path = write_tokens(tmp_path / "tokens.bin", config.vocab_size)
    model, optimizer, train_data, validation_data = training_state(config, token_path)
    payload = make_payload(config, model, optimizer, train_data, validation_data, update_index=0)
    path = tmp_path / "latest.pt"
    save_durable_checkpoint(path, payload)
    assert verify_checkpoint(path).ok

    # A single flipped byte in the middle of the artifact.
    raw = bytearray(path.read_bytes())
    midpoint = len(raw) // 2
    raw[midpoint] ^= 0xFF
    path.write_bytes(bytes(raw))
    report = verify_checkpoint(path)
    assert not report.ok
    assert report.result("checkpoint.checksum").status == "FAIL"
    assert CHECKPOINT_CHECKSUM_MISMATCH in report.result("checkpoint.checksum").reason
    with pytest.raises(CheckpointIntegrityError):
        load_verified_checkpoint(path)

    # A truncated artifact.
    truncated = tmp_path / "truncated.pt"
    save_durable_checkpoint(truncated, payload)
    original = truncated.read_bytes()
    truncated.write_bytes(original[: len(original) // 2])
    truncated_report = verify_checkpoint(truncated)
    assert not truncated_report.ok
    assert truncated_report.result("checkpoint.byte_size").status == "FAIL"

    # No manifest means unverified, never "fine".
    unmanifested = tmp_path / "no_manifest.pt"
    save_durable_checkpoint(unmanifested, payload)
    manifest_path_for(unmanifested).unlink()
    missing_report = verify_checkpoint(unmanifested)
    assert not missing_report.ok
    assert CHECKPOINT_MANIFEST_MISSING in missing_report.result("checkpoint.manifest_present").reason
    with pytest.raises(CheckpointIntegrityError, match=CHECKPOINT_MANIFEST_MISSING):
        read_manifest(unmanifested)


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_mismatched_run_id_frozen_artifacts_and_schedule_are_rejected(tmp_path: Path) -> None:
    config = tiny_config()
    token_path = write_tokens(tmp_path / "tokens.bin", config.vocab_size)
    model, optimizer, train_data, validation_data = training_state(config, token_path)
    payload = make_payload(config, model, optimizer, train_data, validation_data, update_index=0)
    path = tmp_path / "latest.pt"
    save_durable_checkpoint(path, payload)

    drifted = {**artifact_hashes(config), "recipe_digest": "0" * 64}
    report = verify_checkpoint(
        path,
        expected_run_id="run-someothrlineage",
        expected_frozen_config_hashes=drifted,
        expected_schedule_content_hash="a" * 64,
    )
    assert report.result("checkpoint.run_id").status == "FAIL"
    assert report.result("checkpoint.frozen_config_hashes").status == "FAIL"
    assert report.result("checkpoint.schedule_binding").status == "FAIL"
    with pytest.raises(CheckpointIntegrityError):
        load_verified_checkpoint(path, expected_run_id="run-someothrlineage")

    with pytest.raises(TrainingIntegrityError, match="RECIPE_HASH_DRIFT"):
        assert_resume_matches_frozen_artifacts(payload, drifted)
    assert assert_resume_matches_frozen_artifacts(payload, artifact_hashes(config)) is None


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_a_cursorless_payload_may_not_claim_a_schedule(tmp_path: Path) -> None:
    config = tiny_config()
    token_path = write_tokens(tmp_path / "tokens.bin", config.vocab_size)
    model, optimizer, train_data, validation_data = training_state(config, token_path)
    with pytest.raises(CheckpointContractError, match="SCHEDULE_BINDING_MISMATCH"):
        make_payload(
            config,
            model,
            optimizer,
            train_data,
            validation_data,
            update_index=0,
            schedule_content_hash="b" * 64,
        )


# --------------------------------------------------------------------------------------
# Interruption and resume
# --------------------------------------------------------------------------------------


# **Validates: Requirements 2.1, 2.4, 3.1, 3.3**
def test_interruption_and_resume_reproduce_the_next_updates_exactly(tmp_path: Path) -> None:
    config = tiny_config()
    token_path = write_tokens(tmp_path / "tokens.bin", config.vocab_size)
    model, optimizer, train_data, validation_data = training_state(config, token_path)
    scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=4096.0)

    for step in range(2):
        take_update(model, optimizer, train_data)
    payload = make_payload(
        config,
        model,
        optimizer,
        train_data,
        validation_data,
        update_index=1,
        scaler=scaler,
        best_validation=BestValidationState(loss=1.25, update_index=1, relative_path="best.pt"),
    )
    path = tmp_path / "latest.pt"
    save_durable_checkpoint(path, payload)

    expected_losses = [take_update(model, optimizer, train_data) for _ in range(3)]
    expected_weights = state_dict_sha256(model.state_dict())

    # Interruption: a completely fresh process-equivalent state.
    resumed_model, resumed_optimizer, resumed_train, resumed_validation = training_state(
        config, token_path
    )
    resumed_scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=65_536.0)
    loaded = load_verified_checkpoint(
        path,
        expected_run_id="run-fixture0000000",
        expected_frozen_config_hashes=artifact_hashes(config),
        expected_schedule_content_hash=PILOT_SCHEDULE_HASH_SENTINEL,
    )
    resume = restore_durable_state(
        loaded,
        resumed_model,
        resumed_optimizer,
        resumed_train,
        resumed_validation,
        scaler=resumed_scaler,
    )

    assert resume.first_update_index == 2
    assert resume.counters is not None and resume.counters.consumed_loss_tokens == 64
    assert resume.counters.next_update_index == 2
    assert resume.reproducible is True
    assert resume.scaler_restored is True
    assert resumed_scaler.get_scale() == 4096.0
    assert resume.best_validation_loss == 1.25
    assert resume.run_id == "run-fixture0000000"
    assert resume.schedule_cursor is None

    resumed_losses = [take_update(resumed_model, resumed_optimizer, resumed_train) for _ in range(3)]
    assert resumed_losses == expected_losses
    assert state_dict_sha256(resumed_model.state_dict()) == expected_weights


# **Validates: Requirements 3.1, 3.3**
def test_legacy_format_v2_checkpoint_still_restores_through_the_durable_path(tmp_path: Path) -> None:
    """Preservation: the proven format-v2 RNG/sampler resume keeps working unchanged."""
    config = tiny_config()
    token_path = write_tokens(tmp_path / "tokens.bin", config.vocab_size)
    model, optimizer, train_data, validation_data = training_state(config, token_path)
    take_update(model, optimizer, train_data)

    legacy_path = tmp_path / "legacy.pt"
    save_checkpoint(
        legacy_path,
        model,
        optimizer,
        config,
        argparse.Namespace(seed=SEED),
        train_data,
        validation_data,
        step=0,
        best_validation_loss=3.5,
    )
    expected_losses = [take_update(model, optimizer, train_data) for _ in range(2)]

    resumed_model, resumed_optimizer, resumed_train, resumed_validation = training_state(
        config, token_path
    )
    legacy = torch.load(legacy_path, map_location="cpu", weights_only=False)
    resume = restore_durable_state(
        legacy, resumed_model, resumed_optimizer, resumed_train, resumed_validation
    )

    assert resume.first_update_index == 1
    assert resume.best_validation_loss == 3.5
    assert resume.reproducible is True
    # A legacy artifact has no durable fields, and the resume state says so plainly.
    assert resume.scaler_restored is False
    assert resume.counters is None
    assert resume.run_id is None
    assert [take_update(resumed_model, resumed_optimizer, resumed_train) for _ in range(2)] == expected_losses


# --------------------------------------------------------------------------------------
# Retention
# --------------------------------------------------------------------------------------


def recovery(name: str, update_index: int, *, verified: bool = True) -> CheckpointEntry:
    return CheckpointEntry(name, ROLE_RECOVERY, update_index, verified=verified)


# **Validates: Requirements 2.1, 2.4, 3.1**
def test_retention_keeps_latest_three_recovery_states_and_all_protected_roles() -> None:
    entries = [
        recovery("update_100.pt", 100),
        recovery("update_200.pt", 200),
        recovery("update_300.pt", 300),
        recovery("update_400.pt", 400),
        recovery("update_500.pt", 500),
        CheckpointEntry("parent.pt", ROLE_BRANCH_PARENT, 50, verified=True),
        CheckpointEntry("endpoint.pt", ROLE_SELECTED_ENDPOINT, 60, verified=True),
        CheckpointEntry("fallback.pt", ROLE_FALLBACK, 70, verified=True),
    ]
    plan = plan_retention(entries)

    assert set(plan.delete_paths) == {"update_100.pt", "update_200.pt"}
    assert set(plan.keep_paths) == {
        "update_300.pt",
        "update_400.pt",
        "update_500.pt",
        "parent.pt",
        "endpoint.pt",
        "fallback.pt",
    }
    assert plan.decision("parent.pt").action == KEEP
    assert plan.decision("update_100.pt").action == DELETE
    assert retention_plan_violations(plan, entries) == ()
    assert_retention_plan_safe(plan, entries)


# **Validates: Requirements 2.1, 2.4, 3.1**
def test_retention_never_deletes_an_unverified_copy() -> None:
    entries = [
        recovery("update_100.pt", 100, verified=False),
        recovery("update_200.pt", 200, verified=False),
        recovery("update_300.pt", 300),
        recovery("update_400.pt", 400),
        recovery("update_500.pt", 500),
    ]
    plan = plan_retention(entries)
    assert plan.delete_paths == ()
    assert plan.decision("update_100.pt").reason_code == CHECKPOINT_RETENTION_UNVERIFIED_RETAINED

    # With no verified retained successor, nothing is superseded, so nothing is deleted.
    unverified_latest = [
        recovery("update_100.pt", 100),
        recovery("update_300.pt", 300, verified=False),
        recovery("update_400.pt", 400, verified=False),
        recovery("update_500.pt", 500, verified=False),
    ]
    conservative = plan_retention(unverified_latest)
    assert conservative.delete_paths == ()


# **Validates: Requirements 2.1, 2.4, 3.1, 3.2**
@given(
    inventory=st.lists(
        st.tuples(
            st.sampled_from(CHECKPOINT_ROLES),
            st.integers(min_value=0, max_value=40),
            st.booleans(),
        ),
        min_size=1,
        max_size=12,
        unique_by=lambda item: item[1],
    ),
    keep_latest=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=40, deadline=None, derandomize=True)
def test_retention_plan_is_never_unsafe(
    inventory: list[tuple[str, int, bool]], keep_latest: int
) -> None:
    """Property: a proposed deletion is always verified, superseded, and unprotected."""
    entries = [
        CheckpointEntry(f"update_{index}.pt", role, index, verified=verified)
        for role, index, verified in inventory
    ]
    plan = plan_retention(entries, keep_latest=keep_latest)

    assert retention_plan_violations(plan, entries) == ()
    assert len(plan.decisions) == len(entries)
    by_path = {entry.path: entry for entry in entries}
    for decision in plan.delete:
        entry = by_path[decision.path]
        assert entry.role not in PROTECTED_ROLES
        assert entry.verified
    for entry in entries:
        if entry.role in PROTECTED_ROLES or not entry.verified:
            assert plan.decision(entry.path).action == KEEP
    recovery_entries = sorted(
        (entry for entry in entries if entry.role == ROLE_RECOVERY),
        key=lambda entry: -entry.update_index,
    )
    for entry in recovery_entries[:keep_latest]:
        assert plan.decision(entry.path).action == KEEP


# **Validates: Requirements 2.1, 2.4, 2.5**
def test_inventory_marks_an_unverifiable_artifact_and_retention_retains_it(tmp_path: Path) -> None:
    config = tiny_config()
    token_path = write_tokens(tmp_path / "tokens.bin", config.vocab_size)
    model, optimizer, train_data, validation_data = training_state(config, token_path)
    run_dir = tmp_path / "run"
    for index in range(2):
        payload = make_payload(
            config, model, optimizer, train_data, validation_data, update_index=index
        )
        save_durable_checkpoint(run_dir / f"update_{index}.pt", payload)
        take_update(model, optimizer, train_data)
    # Publish one artifact whose manifest never landed, as an interrupted write would.
    payload = make_payload(config, model, optimizer, train_data, validation_data, update_index=2)
    save_durable_checkpoint(run_dir / "update_2.pt", payload)
    manifest_path_for(run_dir / "update_2.pt").unlink()

    entries = inventory_from_directory(run_dir)
    verified = {entry.path: entry.verified for entry in entries}
    assert verified == {"update_0.pt": True, "update_1.pt": True, "update_2.pt": False}
    assert {entry.path: entry.update_index for entry in entries}["update_1.pt"] == 1

    plan = plan_retention(entries, keep_latest=1)
    assert plan.decision("update_2.pt").action == KEEP
    assert plan.decision("update_2.pt").reason_code == CHECKPOINT_RETENTION_UNVERIFIED_RETAINED
    assert retention_plan_violations(plan, entries) == ()


# **Validates: Requirements 2.4, 2.5**
def test_manifest_is_json_and_records_no_fabricated_measurement(tmp_path: Path) -> None:
    config = tiny_config()
    token_path = write_tokens(tmp_path / "tokens.bin", config.vocab_size)
    model, optimizer, train_data, validation_data = training_state(config, token_path)
    payload = make_payload(config, model, optimizer, train_data, validation_data, update_index=0)
    path = tmp_path / "latest.pt"
    save_durable_checkpoint(path, payload)

    recorded = json.loads(manifest_path_for(path).read_text(encoding="utf-8"))
    assert recorded["sha256"] == read_manifest(path).sha256
    assert recorded["remote_copy_verified"].startswith("NOT_RUN")
    # No measured checkpoint wall time or throughput is claimed anywhere in the manifest.
    assert not any("seconds" in key or "throughput" in key for key in recorded)


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_a_failed_load_test_never_publishes_an_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rename is gated by the independent load test, so a bad write is never acknowledged."""
    config = tiny_config()
    token_path = write_tokens(tmp_path / "tokens.bin", config.vocab_size)
    model, optimizer, train_data, validation_data = training_state(config, token_path)
    payload = make_payload(config, model, optimizer, train_data, validation_data, update_index=0)
    path = tmp_path / "run" / "latest.pt"

    def partial_save(obj: object, handle: object) -> None:
        handle.write(b"partially flushed bytes, not a complete archive")  # type: ignore[attr-defined]

    monkeypatch.setattr(torch, "save", partial_save)
    with pytest.raises(CheckpointIntegrityError, match="CHECKPOINT_LOAD_TEST_FAILED"):
        save_durable_checkpoint(path, payload)

    assert not path.exists()
    assert not manifest_path_for(path).exists()
    # The temporary file is removed, so no half-written artifact is left to be mistaken for one.
    assert list(path.parent.iterdir()) == []


class CursorTokenStream:
    """A bounded batch source driven by the real :class:`ScheduleCursor`.

    Task 3.10 already proves the full :class:`~tinybench_lm.schedule.ScheduledTokenStream` over
    source-tagged shards. What matters here is the checkpoint side of that contract: the one
    integer cursor and the schedule content hash it is bound to must survive a save and a
    resume, and a resume against a different schedule must fail closed. This reader uses the
    same cursor object and the same ``state_dict`` surface, over one tiny uint16 file.
    """

    def __init__(self, path: Path, content_hash: str, sequence_length: int) -> None:
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        stride = sequence_length + 1
        self.stride = stride
        self.cursor = ScheduleCursor(content_hash, 0, len(self.tokens) // stride)

    def state_dict(self) -> dict[str, object]:
        return self.cursor.state_dict()

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.cursor.load_state_dict(state)

    def get_batch(self, batch_size: int, seq_len: int, device: torch.device):
        rows = []
        for _ in range(batch_size):
            start = self.cursor.position * self.stride
            rows.append(np.asarray(self.tokens[start : start + self.stride]))
            self.cursor.advance(1)
        batch = torch.from_numpy(np.stack(rows).astype(np.int64, copy=False)).to(device)
        return batch[:, :-1], batch[:, 1:]

    def close(self) -> None:
        mapping = getattr(self.tokens, "_mmap", None)
        if mapping is not None:
            mapping.close()


# **Validates: Requirements 1.1, 2.1, 2.4, 3.1**
def test_schedule_cursor_and_hash_survive_a_save_and_a_verified_resume(tmp_path: Path) -> None:
    config = tiny_config()
    token_path = write_tokens(tmp_path / "tokens.bin", config.vocab_size)
    schedule_hash = "c" * 64
    model = build_initial_model(config, SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_data = CursorTokenStream(token_path, schedule_hash, PLAN.sequence_length)
    validation_data = CursorTokenStream(token_path, schedule_hash, PLAN.sequence_length)
    resumed_train = CursorTokenStream(token_path, schedule_hash, PLAN.sequence_length)
    resumed_validation = CursorTokenStream(token_path, schedule_hash, PLAN.sequence_length)
    other_schedule = CursorTokenStream(token_path, "d" * 64, PLAN.sequence_length)

    try:
        for _ in range(2):
            take_update(model, optimizer, train_data)
        # Two updates of two microbatches of two sequences each.
        assert train_data.cursor.position == 8

        payload = make_payload(
            config,
            model,
            optimizer,
            train_data,
            validation_data,
            update_index=1,
            schedule_content_hash=schedule_hash,
        )
        assert payload["schedule_cursor"] == 8
        assert payload["schedule_content_hash"] == schedule_hash
        assert payload_violations(payload) == ()

        path = tmp_path / "latest.pt"
        save_durable_checkpoint(path, payload)
        assert read_manifest(path).schedule_cursor == 8

        expected_losses = [take_update(model, optimizer, train_data) for _ in range(2)]

        loaded = load_verified_checkpoint(path, expected_schedule_content_hash=schedule_hash)
        resumed_model = build_initial_model(config, SEED)
        resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
        resume = restore_durable_state(
            loaded, resumed_model, resumed_optimizer, resumed_train, resumed_validation
        )
        assert resume.schedule_cursor == 8
        assert resume.schedule_content_hash == schedule_hash
        assert resumed_train.cursor.position == 8
        assert [
            take_update(resumed_model, resumed_optimizer, resumed_train) for _ in range(2)
        ] == expected_losses

        # A resume against a different schedule would silently change the exposure order.
        with pytest.raises(ScheduleResumeError):
            restore_durable_state(
                loaded, build_initial_model(config, SEED), resumed_optimizer, other_schedule, resumed_validation
            )
    finally:
        for stream in (train_data, validation_data, resumed_train, resumed_validation, other_schedule):
            stream.close()
