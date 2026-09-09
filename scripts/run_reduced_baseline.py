"""Prepare, verify, launch, and export the reduced baseline.

The command is intentionally small: preparation is read-only and emits a plan, launch
delegates to ``train.py``, verification checks the durable source endpoint, and export is
allowed only for the completed decayed endpoint.  It never deletes checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from tinybench_lm.checkpointing import frozen_config_hashes, verify_checkpoint
from tinybench_lm.exposure import BASELINE_RECIPE_SHA256, load_exposure_plan, verify_exposure
from tinybench_lm.evaluation_protocol import load_evaluation_protocol
from tinybench_lm.provenance import export_release, read_step_zero_provenance, verify_release_export, verify_step_zero_provenance
from tinybench_lm.schedule import load_schedule
from tinybench_lm.shards import load_split_manifest
from tinybench_lm.training_recipe import (
    RunSemantics,
    WSDSchedule,
    adamw_settings,
    load_training_recipe,
    model_config_hash,
    recipe_digest,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/training/baseline_reduced_v1.yaml"
EXPOSURE_ROOT = ROOT / "runs/reduced_campaign/reduced_baseline_v1"
DEV_MANIFEST = ROOT / "data/shards/reduced_5pct_v1/validation_dev.manifest.json"
DEV_SCHEDULE = ROOT / "data/schedules/reduced_5pct_v1/validation_dev.json"
DEFAULT_RUN_DIR = ROOT / "runs/reduced_campaign/reduced_baseline_v1/run"


class RunnerError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _norm_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _verified_baseline_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Trust the registered bytes, then reject substitutions of their parsed semantics."""
    content = CONFIG.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(content).hexdigest() != BASELINE_RECIPE_SHA256:
        raise RunnerError("baseline_reduced_v1 does not match its trusted frozen digest; publish a successor")
    frozen = yaml.safe_load(content.decode("utf-8"))
    if config is not None and dict(config) != frozen:
        raise RunnerError("baseline config differs from trusted frozen semantics; publish a successor")
    return frozen


def _baseline_evaluation_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = config["interfaces"]["evaluation_binding"]
    protocol = load_evaluation_protocol(ROOT / contract["protocol_path"])
    if protocol["_digest"] != contract["protocol_sha256"]:
        raise RunnerError("evaluation protocol identity does not match the frozen baseline contract")
    return protocol


def _required_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    data = config["data"]
    schedules = config["schedule_inputs"]
    return {
        "model_config": ROOT / config["model"]["config_path"],
        "tokenizer_protocol": ROOT / config["tokenizer"]["protocol_path"],
        "tokenizer": ROOT / config["tokenizer"]["artifact_path"],
        "shard_protocol": ROOT / data["shard_protocol_path"],
        "source_protocol": ROOT / data["source_protocol_path"],
        "decontam_protocol": ROOT / data["decontamination_protocol_path"],
        "base_schedule": ROOT / schedules["base_schedule"]["path"],
        "dev_manifest": DEV_MANIFEST,
        "dev_schedule": DEV_SCHEDULE,
        "exposure_plan": EXPOSURE_ROOT / "exposure_plan.json",
        "component_1": EXPOSURE_ROOT / "component_1.json",
        "component_2": EXPOSURE_ROOT / "component_2.json",
    }


def build_identity(*, config: Mapping[str, Any], paths: Mapping[str, Path], environment: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build the runner-owned identity, including policy fields omitted by legacy RunSemantics."""
    config = _verified_baseline_config(config)
    evaluation_protocol = _baseline_evaluation_protocol(config)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RunnerError(f"baseline input is missing: {missing}")
    exposure = load_exposure_plan(paths["exposure_plan"], (paths["component_1"], paths["component_2"]))
    stable_manifest = load_split_manifest(ROOT / config["data"]["manifests"]["stable_train"]["path"])
    verify_exposure(exposure, stable_manifest)
    dev_manifest = load_split_manifest(paths["dev_manifest"])
    dev_schedule = load_schedule(paths["dev_schedule"])
    if dev_manifest.split_id != "validation_dev" or dev_schedule.split_id != "validation_dev":
        raise RunnerError("development validation inputs must be validation_dev")
    from tinybench_lm.evaluation_binding import resolve_effective_binding
    _, evaluation_facts = resolve_effective_binding(
        evaluation_protocol, ["hellaswag", "arc_easy", "piqa", "winogrande", "wikitext103"]
    )
    expected = {
        # The contract sidecar contains a historical single-nibble typo; bind the actual
        # frozen file identity recorded by the repository inventory.
        "model_config": "69db94fe2bd0fef99b7a8ee183678b3bc2fc8098a9cdea8fd71adf1149bb1782",
        "tokenizer_protocol": config["tokenizer"]["protocol_sha256_normalized_lf"],
        "tokenizer": config["tokenizer"]["artifact_sha256"],
        "shard_protocol": config["data"]["shard_protocol_sha256_normalized_lf"],
        "source_protocol": config["data"]["source_protocol_sha256_normalized_lf"],
        "decontam_protocol": config["data"]["decontamination_protocol_sha256_normalized_lf"],
        "dev_manifest": config["data"]["manifests"]["validation_dev"]["file_sha256"],
        "dev_schedule": config["schedule_inputs"]["development_schedule"]["file_sha256"],
    }
    for name, digest in expected.items():
        observed = sha256(paths[name]) if name in {"model_config", "tokenizer", "dev_manifest", "dev_schedule"} else _norm_hash(paths[name])
        if observed != digest:
            raise RunnerError(f"{name} identity does not match the frozen baseline contract")
    expected_dev_content = config["schedule_inputs"]["development_schedule"]["content_hash"]
    if dev_schedule.content_hash() != expected_dev_content:
        raise RunnerError("development schedule content identity does not match the frozen contract")
    if sha256(paths["exposure_plan"]) != "76f16bfc98620227e2070645e5e901d8a4a8811c3970509b92769ebd84f71f8f":
        raise RunnerError("exposure plan file identity does not match the verified baseline artifact")
    schedule = config["learning_rate"]
    identity = {
        "identity_schema": "reduced_baseline_runner_v1",
        "scope": "baseline_reduced_v1",
        "baseline_contract_sha256_normalized_lf": BASELINE_RECIPE_SHA256,
        "evaluation_protocol_sha256_normalized_lf": evaluation_protocol["_digest"],
        "recipe_sha256_normalized_lf": config["optimizer"]["recipe_sha256_normalized_lf"],
        "model_config_sha256": _norm_hash(paths["model_config"]),
        "model_config_canonical_hash": str(config["model"]["canonical_config_hash"]),
        "tokenizer_protocol_sha256": _norm_hash(paths["tokenizer_protocol"]),
        "tokenizer_artifact_sha256": sha256(paths["tokenizer"]),
        "shard_protocol_sha256": _norm_hash(paths["shard_protocol"]),
        "source_protocol_sha256": _norm_hash(paths["source_protocol"]),
        "decontam_protocol_sha256": _norm_hash(paths["decontam_protocol"]),
        "exposure_plan_hash": sha256(paths["exposure_plan"]),
        "exposure_content_hash": exposure.content_hash,
        "exposure_component_hashes": [component.content_hash() for component in exposure.components],
        "validation_manifest_hash": dev_manifest.content_hash(),
        "validation_schedule_hash": dev_schedule.content_hash(),
        "validation_schedule_id": dev_schedule.schedule_id,
        "validation_mode": "full-dev",
        "eval_interval": 100,
        "save_interval": 100,
        "train_schedule_content_hash": exposure.content_hash,
        "seed": int(config["seed"]["value"]),
        "total_updates": int(config["horizon"]["total_updates"]),
        "consumed_loss_tokens": int(config["horizon"]["consumed_loss_tokens"]),
        "loss_tokens_per_update": int(config["horizon"]["loss_tokens_per_update"]),
        "micro_batch_size": int(config["batch"]["micro_batch_size"]),
        "gradient_accumulation": int(config["batch"]["gradient_accumulation"]),
        "sequence_length": int(config["batch"]["sequence_length"]),
        "peak_lr": float(schedule["peak_lr"]),
        "warmup_updates": int(schedule["warmup_updates"]),
        "stable_updates": int(schedule["stable_updates"]),
        "decay_updates": int(schedule["decay_updates"]),
        "weight_decay": float(config["optimizer"]["weight_decay"]),
        "gradient_clip_global_norm": float(config["optimizer"]["gradient_clip_global_norm"]),
        "precision": {"dtype": "bfloat16", "grad_scaler": False},
        "determinism_policy": config["precision"]["determinism_policy"],
        "evaluation_binding": evaluation_facts["binding_digest"],
        "evaluation_binding_facts": evaluation_facts,
        "environment": dict(environment or {}),
    }
    identity["run_id"] = "baseline-" + hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    identity["exposure"] = exposure.to_dict()
    return identity


def prepare(
    *,
    config_path: Path = CONFIG,
    environment: Mapping[str, Any] | None = None,
    run_dir: Path = DEFAULT_RUN_DIR,
) -> dict[str, Any]:
    if config_path.resolve() != CONFIG.resolve():
        raise RunnerError("production baseline runner is bound to baseline_reduced_v1.yaml")
    config = _verified_baseline_config()
    paths = _required_paths(config)
    identity = build_identity(config=config, paths=paths, environment=environment)
    needed = int(config["horizon"]["consumed_sequences"])
    actual = int(identity["exposure"]["sequence_count"])
    if actual < needed:
        raise RunnerError(f"exposure supplies {actual} sequences, needs {needed}")
    identity["paths"] = {key: str(path.relative_to(ROOT)) for key, path in paths.items()}
    identity["command"] = launch_command(identity, run_dir=run_dir)
    return identity


def launch_command(
    identity: Mapping[str, Any],
    *,
    run_dir: Path,
    resume: Path | None = None,
    stop_after_updates: int | None = None,
    stop_after_training_seconds: float | None = None,
) -> list[str]:
    paths = identity["paths"]
    command = [sys.executable, "train.py", "--config", paths["model_config"], "--run-dir", str(run_dir),
            "--shard-root", "data/shards/reduced_5pct_v1", "--train-manifest", "data/shards/reduced_5pct_v1/stable_train.manifest.json",
            "--validation-manifest", paths["dev_manifest"], "--validation-schedule", paths["dev_schedule"],
            "--exposure-plan", paths["exposure_plan"], "--exposure-component-1", paths["component_1"], "--exposure-component-2", paths["component_2"],
            "--runner-identity", str(run_dir / "runner_identity.json"),
            "--steps", str(identity["total_updates"]), "--micro-batch-size", str(identity["micro_batch_size"]),
            "--sequence-length", str(identity["sequence_length"]), "--gradient-accumulation", str(identity["gradient_accumulation"]),
            "--learning-rate", str(identity["peak_lr"]), "--warmup-steps", str(identity["warmup_updates"]), "--decay-updates", str(identity["decay_updates"]),
            "--bf16-stability", "stable", "--validation-mode", "full-dev", "--eval-interval", "100", "--save-interval", "100"]
    if resume is not None:
        command.extend(["--resume", str(resume)])
    if stop_after_updates is not None:
        if not 0 < int(stop_after_updates) <= int(identity["total_updates"]):
            raise RunnerError("stop-after-updates must be positive and within the fixed horizon")
        command.extend(["--stop-after-updates", str(int(stop_after_updates))])
    if stop_after_training_seconds is not None:
        if float(stop_after_training_seconds) <= 0:
            raise RunnerError("stop-after-training-seconds must be positive")
        command.extend(["--stop-after-training-seconds", str(float(stop_after_training_seconds))])
    return command


def assert_resume_stop_advances(resume: Path, stop_after_updates: int | None) -> None:
    """Prevent an absolute bounded stop from overrunning on the first resumed update."""
    if stop_after_updates is None:
        return
    report = verify_checkpoint(resume)
    if not report.ok:
        raise RunnerError("resume checkpoint failed durable verification")
    payload = torch.load(resume, map_location="cpu", weights_only=False)
    completed = int(payload.get("counters", {}).get("updates_completed", -1))
    if int(stop_after_updates) <= completed:
        raise RunnerError(
            f"stop-after-updates is absolute and must exceed the resumed {completed} completed updates"
        )


def verify_source(checkpoint: Path, *, identity: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise RunnerError(f"checkpoint is missing: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    runner_identity_path = run_dir / "runner_identity.json"
    if not runner_identity_path.is_file():
        raise RunnerError("runner identity manifest is missing")
    recorded_identity = json.loads(runner_identity_path.read_text(encoding="utf-8"))
    if recorded_identity.get("run_id") != identity.get("run_id"):
        raise RunnerError("runner identity drifted; refusing resume/export")
    recorded_identity_hash = sha256(runner_identity_path)
    if payload.get("training_args", {}).get("runner_identity_hash") != recorded_identity_hash:
        raise RunnerError("durable checkpoint is not bound to the runner identity manifest")
    counters = payload.get("counters", {})
    expected_updates = int(identity["total_updates"])
    expected_tokens = int(identity["consumed_loss_tokens"])
    if int(counters.get("updates_completed", -1)) != expected_updates:
        raise RunnerError("source checkpoint is early-stopped and is not export eligible")
    if int(counters.get("consumed_loss_tokens", -1)) != expected_tokens:
        raise RunnerError("source checkpoint consumed-loss counter does not match the fixed horizon")
    if int(payload.get("schedule_cursor", -1)) != int(identity["exposure"]["sequence_count"]):
        raise RunnerError("source checkpoint cursor is not the completed composite exposure cursor")
    args = payload.get("training_args", {})
    if args.get("validation_mode") != "full-dev" or int(args.get("eval_interval", -1)) != 100 or int(args.get("save_interval", -1)) != 100:
        raise RunnerError("source checkpoint validation/recovery policy is not the fixed baseline policy")
    expected_args = {
        "steps": expected_updates, "seed": int(identity["seed"]), "learning_rate": float(identity["peak_lr"]),
        "warmup_steps": int(identity["warmup_updates"]), "decay_updates": int(identity["decay_updates"]),
        "micro_batch_size": int(identity["micro_batch_size"]), "gradient_accumulation": int(identity["gradient_accumulation"]),
        "sequence_length": int(identity["sequence_length"]),
        "weight_decay": float(identity["weight_decay"]), "grad_clip": float(identity["gradient_clip_global_norm"]),
    }
    for name, expected_value in expected_args.items():
        actual = args.get(name)
        if isinstance(expected_value, float):
            if actual is None or abs(float(actual) - expected_value) > 1e-12:
                raise RunnerError(f"source checkpoint training argument {name} drifted")
        elif int(actual) != expected_value:
            raise RunnerError(f"source checkpoint training argument {name} drifted")
    schedule = WSDSchedule(expected_updates, int(identity["warmup_updates"]), int(identity["decay_updates"]), float(identity["peak_lr"]))
    if schedule.learning_rate(expected_updates - 1) != 0.0:
        raise RunnerError("source checkpoint uses a nonzero final learning rate")
    if int(payload.get("step", -1)) != expected_updates - 1:
        raise RunnerError("source checkpoint step does not identify the completed endpoint")
    if any(float(group.get("lr", 1.0)) != 0.0 for group in payload.get("optimizer", {}).get("param_groups", ())):
        raise RunnerError("source checkpoint optimizer still carries a nonzero applied learning rate")
    payload_model_hash = model_config_hash(payload.get("model_config", {}))
    if payload_model_hash != identity.get("model_config_canonical_hash"):
        raise RunnerError("source checkpoint model identity does not match the runner manifest")
    recipe = load_training_recipe()
    settings = adamw_settings(recipe)
    expected_semantics = RunSemantics(
        recipe_digest=recipe_digest(recipe),
        model_config_hash=payload_model_hash,
        peak_lr=float(identity["peak_lr"]),
        total_updates=expected_updates,
        warmup_updates=int(identity["warmup_updates"]),
        decay_updates=int(identity["decay_updates"]),
        loss_tokens_per_update=int(identity["loss_tokens_per_update"]),
        weight_decay=float(identity["weight_decay"]),
        beta1=float(settings["betas"][0]), beta2=float(settings["betas"][1]),
        epsilon=float(settings["epsilon"]),
        gradient_clip_global_norm=float(identity["gradient_clip_global_norm"]),
        precision_dtype=str(identity["precision"]["dtype"]),
        grad_scaler=bool(identity["precision"]["grad_scaler"]),
        seed=int(identity["seed"]),
        train_schedule_content_hash=str(identity["exposure_content_hash"]),
    )
    expected_training_run_id = expected_semantics.run_id(recipe)
    expected_frozen = frozen_config_hashes(model_config_hash=payload_model_hash, recipe=recipe)
    provenance_path = run_dir / "step_zero_provenance.json"
    if not provenance_path.is_file():
        raise RunnerError("fresh step-zero provenance is missing")
    provenance = read_step_zero_provenance(provenance_path)
    if provenance.seed != int(identity["seed"]) or provenance.model_config != payload.get("model_config"):
        raise RunnerError("step-zero provenance is not bound to the source model or seed")
    provenance_report = verify_step_zero_provenance(provenance, reproduce=False)
    if not provenance_report.ok:
        raise RunnerError("step-zero provenance verification failed")
    report = verify_checkpoint(
        checkpoint,
        expected_run_id=expected_training_run_id,
        expected_frozen_config_hashes=expected_frozen,
        expected_schedule_content_hash=identity["exposure_content_hash"],
    )
    if not report.ok:
        raise RunnerError("durable checkpoint verification failed: " + "; ".join(result.reason for result in report.failures))
    return {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "run_id": payload["run_id"], "updates_completed": expected_updates, "consumed_loss_tokens": expected_tokens, "schedule_cursor": int(payload["schedule_cursor"]), "final_learning_rate": 0.0}


def export_completed(*, checkpoint: Path, destination: Path, identity: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    source = verify_source(checkpoint, identity=identity, run_dir=run_dir)
    provenance = read_step_zero_provenance(run_dir / "step_zero_provenance.json")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    from tinybench_lm import ModelConfig, TinyBenchLM
    model = TinyBenchLM(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model"])
    export_payload = export_release(destination, model, provenance=provenance, notes="completed-decay reduced baseline source checkpoint")
    report = verify_release_export(destination, expected_parameter_count=export_payload["unique_parameter_count"])
    if not report.ok:
        raise RunnerError("export reload verification failed")
    evidence = {"status": "PASS", "source": source, "export_sha256": sha256(destination), "destination": str(destination), "run_identity": identity["run_id"]}
    destination.with_suffix(destination.suffix + ".evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("prepare", "launch", "verify", "export"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path, help="verified durable checkpoint to resume for launch")
    parser.add_argument("--stop-after-updates", type=int, help="bounded rehearsal stop at an update boundary")
    parser.add_argument("--stop-after-training-seconds", type=float, help="bounded optimizer-time rehearsal stop")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--execute", action="store_true", help="launch training after printing the exact command")
    args = parser.parse_args()
    identity = prepare(
        config_path=args.config,
        environment={"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda},
        run_dir=args.run_dir,
    )
    if args.command == "prepare":
        print(json.dumps(identity, indent=2, sort_keys=True))
        return
    if args.command == "launch":
        command = launch_command(
            identity,
            run_dir=args.run_dir,
            resume=args.resume,
            stop_after_updates=args.stop_after_updates,
            stop_after_training_seconds=args.stop_after_training_seconds,
        )
        if args.execute:
            if identity["evaluation_binding"] == "PENDING_SECTION_5":
                raise RunnerError("baseline launch is blocked until section 5 resolves evaluation binding")
            if args.resume is not None:
                assert_resume_stop_advances(args.resume, args.stop_after_updates)
            args.run_dir.mkdir(parents=True, exist_ok=True)
            identity_path = args.run_dir / "runner_identity.json"
            identity_bytes = (json.dumps(identity, indent=2, sort_keys=True) + "\n").encode()
            if identity_path.exists() and identity_path.read_bytes() != identity_bytes:
                raise RunnerError("existing runner identity differs; refusing to mutate a resume lineage")
            if not identity_path.exists():
                identity_path.write_bytes(identity_bytes)
        print(json.dumps({"run_id": identity["run_id"], "command": command}, indent=2))
        if args.execute:
            raise SystemExit(subprocess.call(command, cwd=ROOT))
        return
    checkpoint = args.checkpoint or (args.run_dir / "completed.pt")
    if args.command == "verify":
        print(json.dumps(verify_source(checkpoint, identity=identity, run_dir=args.run_dir), indent=2, sort_keys=True))
        return
    destination = args.destination or (args.run_dir / "baseline_export.pt")
    print(json.dumps(export_completed(checkpoint=checkpoint, destination=destination, identity=identity, run_dir=args.run_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
