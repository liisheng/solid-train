"""Bounded section-4 runner custody checks; no training or corpus acquisition."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tinybench_lm import ModelConfig, TinyBenchLM
from tinybench_lm.checkpointing import CheckpointCounters, BestValidationState, build_checkpoint_payload, frozen_config_hashes, save_durable_checkpoint
from tinybench_lm.provenance import export_release, verify_release_export, record_step_zero_provenance
from tinybench_lm.training_recipe import BatchPlan, RunSemantics, WSDSchedule, adamw_settings, load_training_recipe, model_config_hash, recipe_digest
from scripts.run_reduced_baseline import RunnerError, assert_resume_stop_advances, export_completed, launch_command, prepare, sha256, verify_source


def identity() -> dict[str, object]:
    return {
        "run_id": "baseline-fixture0000001", "total_updates": 2,
        "consumed_loss_tokens": 32, "loss_tokens_per_update": 16, "seed": 1337, "peak_lr": 0.001,
        "warmup_updates": 0, "decay_updates": 2, "micro_batch_size": 1,
        "gradient_accumulation": 2, "sequence_length": 8,
        "weight_decay": 0.1, "gradient_clip_global_norm": 1.0,
        "validation_mode": "full-dev", "eval_interval": 100, "save_interval": 100,
        "validation_schedule_hash": "fixture-dev", "train_schedule_content_hash": "fixture-exposure",
        "exposure_content_hash": "fixture-exposure",
        "exposure": {"sequence_count": 4},
        "paths": {"model_config": "configs/final_49m.json", "dev_manifest": "dev.manifest", "dev_schedule": "dev.schedule", "exposure_plan": "exposure_plan.json", "component_1": "component_1.json", "component_2": "component_2.json"},
    }


def test_launch_command_binds_composite_and_fixed_cadence(tmp_path: Path) -> None:
    command = launch_command(
        identity(), run_dir=tmp_path, resume=tmp_path / "latest.pt",
        stop_after_updates=1, stop_after_training_seconds=2.5,
    )
    assert "--exposure-plan" in command
    assert command[command.index("--eval-interval") + 1] == "100"
    assert command[command.index("--save-interval") + 1] == "100"
    assert "--runner-identity" in command
    assert command[command.index("--resume") + 1] == str(tmp_path / "latest.pt")
    assert command[command.index("--stop-after-updates") + 1] == "1"
    assert command[command.index("--stop-after-training-seconds") + 1] == "2.5"
    with pytest.raises(RunnerError, match="within the fixed horizon"):
        launch_command(identity(), run_dir=tmp_path, stop_after_updates=3)


def test_verify_source_rejects_early_stopped_checkpoint(tmp_path: Path) -> None:
    ident = identity()
    (tmp_path / "runner_identity.json").write_text(json.dumps(ident, sort_keys=True), encoding="utf-8")
    checkpoint = tmp_path / "early.pt"
    import hashlib
    identity_path = tmp_path / "runner_identity.json"
    runner_hash = hashlib.sha256(identity_path.read_bytes()).hexdigest()
    torch.save({"counters": {"updates_completed": 1, "consumed_loss_tokens": 8}, "training_args": {"runner_identity_hash": runner_hash}}, checkpoint)
    try:
        verify_source(checkpoint, identity=ident, run_dir=tmp_path)
    except RunnerError as error:
        assert "early-stopped" in str(error)
    else:  # pragma: no cover
        raise AssertionError("early checkpoint was accepted")


def test_resumed_bounded_stop_must_advance_past_completed_update(tmp_path: Path) -> None:
    checkpoint, _ = completed_fixture(tmp_path)
    with pytest.raises(RunnerError, match="must exceed the resumed 2"):
        assert_resume_stop_advances(checkpoint, 2)
    assert_resume_stop_advances(checkpoint, 3)


def test_tiny_clean_export_reloads_and_has_no_training_state(tmp_path: Path) -> None:
    config = ModelConfig(vocab_size=32, max_seq_len=8, d_model=16, n_layers=1, n_heads=2, n_kv_heads=1, d_ff=32, dropout=0.0)
    torch.manual_seed(1337)
    model = TinyBenchLM(config)
    destination = tmp_path / "tiny_export.pt"
    provenance = record_step_zero_provenance(model, config, seed=1337, config_path=None)
    export_release(destination, model, provenance=provenance, notes="fixture only")
    report = verify_release_export(destination)
    assert report.ok, report.to_dict()


def test_completed_durable_fixture_passes_runner_verify_and_export(tmp_path: Path) -> None:
    checkpoint, ident = completed_fixture(tmp_path)
    source = verify_source(checkpoint, identity=ident, run_dir=tmp_path)
    assert source["schedule_cursor"] == 4
    evidence = export_completed(checkpoint=checkpoint, destination=tmp_path / "export.pt", identity=ident, run_dir=tmp_path)
    assert evidence["source"]["checkpoint_sha256"] == sha256(checkpoint)
    assert verify_release_export(tmp_path / "export.pt").ok


def completed_fixture(tmp_path: Path, *, cursor: int = 4, optimizer_lr: float = 0.0) -> tuple[Path, dict[str, object]]:
    config = ModelConfig(vocab_size=32, max_seq_len=8, d_model=16, n_layers=1, n_heads=2, n_kv_heads=1, d_ff=32, dropout=0.0)
    torch.manual_seed(1337)
    model = TinyBenchLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_lr)
    plan = BatchPlan(1, 8, 2)

    class Stream:
        def __init__(self): self.state = {"format_version": 1, "schedule_cursor": cursor, "schedule_content_hash": "fixture-exposure"}
        def state_dict(self): return dict(self.state)

    stream = Stream()
    args = {"steps": 2, "seed": 1337, "learning_rate": 0.001, "warmup_steps": 0, "decay_updates": 2,
            "micro_batch_size": 1, "gradient_accumulation": 2, "sequence_length": 8,
            "weight_decay": 0.1, "grad_clip": 1.0,
            "validation_mode": "full-dev", "eval_interval": 100, "save_interval": 100}
    runner_path = tmp_path / "runner_identity.json"
    ident = identity() | {"model_config_canonical_hash": model_config_hash(config.to_dict()), "precision": {"dtype": "float32", "grad_scaler": False}}
    runner_path.write_text(json.dumps(ident, sort_keys=True), encoding="utf-8")
    args["runner_identity_hash"] = sha256(runner_path)
    provenance = record_step_zero_provenance(model, config, seed=1337)
    (tmp_path / "step_zero_provenance.json").write_text(json.dumps(provenance.to_dict(), sort_keys=True), encoding="utf-8")
    recipe = load_training_recipe()
    settings = adamw_settings(recipe)
    semantics = RunSemantics(
        recipe_digest=recipe_digest(recipe), model_config_hash=model_config_hash(config.to_dict()),
        peak_lr=0.001, total_updates=2, warmup_updates=0, decay_updates=2,
        loss_tokens_per_update=16, weight_decay=0.1, beta1=float(settings["betas"][0]),
        beta2=float(settings["betas"][1]), epsilon=float(settings["epsilon"]),
        gradient_clip_global_norm=1.0, precision_dtype="float32", grad_scaler=False,
        seed=1337, train_schedule_content_hash="fixture-exposure",
    )
    payload = build_checkpoint_payload(model=model, optimizer=optimizer, config=config, args=args,
        train_data=stream, validation_data=stream, counters=CheckpointCounters.at_update(1, plan),
        run_id=semantics.run_id(recipe), frozen_config_hashes=frozen_config_hashes(model_config_hash=model_config_hash(config.to_dict()), recipe=recipe),
        schedule_content_hash="fixture-exposure", best_validation=BestValidationState.unevaluated())
    checkpoint = tmp_path / "completed.pt"
    save_durable_checkpoint(checkpoint, payload)
    return checkpoint, ident


def test_verify_source_rejects_wrong_completed_cursor(tmp_path: Path) -> None:
    checkpoint, ident = completed_fixture(tmp_path, cursor=3)
    with pytest.raises(RunnerError, match="cursor"):
        verify_source(checkpoint, identity=ident, run_dir=tmp_path)


def test_verify_source_rejects_nonzero_applied_final_lr(tmp_path: Path) -> None:
    checkpoint, ident = completed_fixture(tmp_path, optimizer_lr=0.001)
    with pytest.raises(RunnerError, match="nonzero applied learning rate"):
        verify_source(checkpoint, identity=ident, run_dir=tmp_path)


def test_verify_source_rejects_provenance_seed_drift(tmp_path: Path) -> None:
    checkpoint, ident = completed_fixture(tmp_path)
    config = ModelConfig(vocab_size=32, max_seq_len=8, d_model=16, n_layers=1, n_heads=2, n_kv_heads=1, d_ff=32, dropout=0.0)
    torch.manual_seed(99)
    wrong = record_step_zero_provenance(TinyBenchLM(config), config, seed=99)
    (tmp_path / "step_zero_provenance.json").write_text(json.dumps(wrong.to_dict(), sort_keys=True), encoding="utf-8")
    with pytest.raises(RunnerError, match="provenance"):
        verify_source(checkpoint, identity=ident, run_dir=tmp_path)


def test_verify_source_rejects_a_checkpoint_from_another_run_identity(tmp_path: Path) -> None:
    checkpoint, ident = completed_fixture(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["run_id"] = "run-from-another-lineage"
    save_durable_checkpoint(checkpoint, payload)
    with pytest.raises(RunnerError, match="durable checkpoint verification failed"):
        verify_source(checkpoint, identity=ident, run_dir=tmp_path)


def test_verify_source_rejects_wrong_frozen_recipe_hash(tmp_path: Path) -> None:
    checkpoint, ident = completed_fixture(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["frozen_config_hashes"]["recipe_digest"] = "0" * 64
    save_durable_checkpoint(checkpoint, payload)
    with pytest.raises(RunnerError, match="durable checkpoint verification failed"):
        verify_source(checkpoint, identity=ident, run_dir=tmp_path)


def test_verify_source_rejects_changed_runner_manifest_after_checkpoint(tmp_path: Path) -> None:
    checkpoint, ident = completed_fixture(tmp_path)
    changed = dict(ident)
    changed["environment"] = {"torch": "different"}
    (tmp_path / "runner_identity.json").write_text(json.dumps(changed, sort_keys=True), encoding="utf-8")
    with pytest.raises(RunnerError, match="not bound to the runner identity"):
        verify_source(checkpoint, identity=changed, run_dir=tmp_path)


def test_production_runner_rejects_an_engineering_contract(tmp_path: Path) -> None:
    engineering = tmp_path / "engineering.yaml"
    engineering.write_text("protocol: engineering_fixture\n", encoding="utf-8")
    with pytest.raises(RunnerError, match="bound to baseline_reduced_v1.yaml"):
        prepare(config_path=engineering)


def test_production_prepare_binds_the_resolved_evaluation_loader_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """The runner must carry the semantic v3 loader binding, never a sentinel."""
    from scripts import run_reduced_baseline as runner

    # This unit test exercises real config validation and loader-binding resolution.
    # Corpus artifacts are deliberately absent from the CPU verification image;
    # their integrity is checked separately by the production-input preflight.
    config = runner._verified_baseline_config()
    paths = runner._required_paths(config)
    artifact_names = {"base_schedule", "dev_manifest", "dev_schedule", "exposure_plan", "component_1", "component_2"}
    artifact_paths = {paths[name] for name in artifact_names}
    real_is_file = Path.is_file
    real_sha256 = runner.sha256
    artifact_hashes = {
        paths["dev_manifest"]: config["data"]["manifests"]["validation_dev"]["file_sha256"],
        paths["dev_schedule"]: config["schedule_inputs"]["development_schedule"]["file_sha256"],
        paths["exposure_plan"]: "76f16bfc98620227e2070645e5e901d8a4a8811c3970509b92769ebd84f71f8f",
    }

    def fixture_is_file(path):
        return path in artifact_paths or real_is_file(path)

    def fixture_sha256(path):
        return artifact_hashes[path] if path in artifact_hashes else real_sha256(path)

    exposure = SimpleNamespace(
        content_hash="fixture-exposure",
        components=(),
        to_dict=lambda: {"sequence_count": config["horizon"]["consumed_sequences"]},
    )
    stable_manifest = SimpleNamespace(split_id="stable_train")
    dev_manifest = SimpleNamespace(split_id="validation_dev", content_hash=lambda: "fixture-dev-manifest")
    dev_schedule = SimpleNamespace(
        split_id="validation_dev", schedule_id="fixture-dev",
        content_hash=lambda: config["schedule_inputs"]["development_schedule"]["content_hash"],
    )

    def fixture_verify_exposure(actual_exposure, actual_manifest):
        assert actual_exposure is exposure
        assert actual_manifest is stable_manifest

    def fixture_load_exposure(plan_path, component_paths):
        assert plan_path == paths["exposure_plan"]
        assert component_paths == (paths["component_1"], paths["component_2"])
        return exposure

    def fixture_load_manifest(path):
        if path == paths["dev_manifest"]:
            return dev_manifest
        assert path == runner.ROOT / config["data"]["manifests"]["stable_train"]["path"]
        return stable_manifest

    def fixture_load_schedule(path):
        assert path == paths["dev_schedule"]
        return dev_schedule

    monkeypatch.setattr(Path, "is_file", fixture_is_file)
    monkeypatch.setattr(runner, "sha256", fixture_sha256)
    monkeypatch.setattr(runner, "load_exposure_plan", fixture_load_exposure)
    monkeypatch.setattr(runner, "load_split_manifest", fixture_load_manifest)
    monkeypatch.setattr(runner, "load_schedule", fixture_load_schedule)
    monkeypatch.setattr(runner, "verify_exposure", fixture_verify_exposure)
    prepared = prepare()
    facts = prepared["evaluation_binding_facts"]
    assert prepared["evaluation_binding"] == facts["binding_digest"]
    assert prepared["evaluation_binding"] != "PENDING_SECTION_5"
    assert facts["decontamination_protocol_sha256"] == prepared["decontam_protocol_sha256"]
    assert tuple(facts["tasks"]) == ("hellaswag", "arc_easy", "piqa", "winogrande", "wikitext103")


def test_train_rejects_runner_manifest_that_disagrees_with_actual_inputs(tmp_path: Path) -> None:
    import train

    config = ModelConfig(vocab_size=32, max_seq_len=8, d_model=16, n_layers=1, n_heads=2, n_kv_heads=1, d_ff=32, dropout=0.0)
    manifest = identity() | {"identity_schema": "reduced_baseline_runner_v1", "model_config_canonical_hash": model_config_hash(config.to_dict()), "precision": {"dtype": "float32", "grad_scaler": False}, "seed": 7}
    path = tmp_path / "runner_identity.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    args = SimpleNamespace(runner_identity=path, validation_mode="full-dev", eval_interval=100, save_interval=100, seed=1337, steps=2)
    plan = BatchPlan(1, 8, 2)
    precision = SimpleNamespace(dtype_name="float32", use_grad_scaler=False)
    facts = {"train_schedule_content_hash": "fixture-exposure", "validation_schedule_content_hash": "fixture-dev"}
    with pytest.raises(ValueError, match="runner identity field seed differs"):
        train.assert_runner_identity(args, config=config, plan=plan, lr_schedule=WSDSchedule(2, 0, 2, 0.001), precision=precision, data_facts=facts)
