"""Run the bounded reduced-G3 integration rehearsal and write machine-readable evidence.

This harness owns only the section-6 engineering rehearsal.  It preserves the production
3,815-update runner identity and uses absolute early-stop targets for two fresh lineages:
an uninterrupted eight-update run and a four-update stop followed by an eight-update resume.
It never exports either early checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# These imports follow the path bootstrap so direct script execution can import scripts.
import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.run_reduced_baseline import prepare, sha256  # noqa: E402
from tinybench_lm.checkpointing import verify_checkpoint  # noqa: E402
from tinybench_lm.eligibility import production_python_paths  # noqa: E402
from tinybench_lm.exposure import load_exposure_plan  # noqa: E402
from tinybench_lm.schedule import training_order_hash  # noqa: E402
DEFAULT_OUTPUT = ROOT / "runs/verification/g3-section6-integration"


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _source_manifest() -> dict[str, Any]:
    files = {
        str(path.relative_to(ROOT)).replace("\\", "/"): {
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in production_python_paths(ROOT)
    }
    return {"schema": "eligible_production_python_manifest_v1", "files": files}


def _run(command: list[str], *, output: Path) -> dict[str, Any]:
    output = output.resolve()
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=True)
    (output / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (output / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    return {
        "command": [str(item) for item in command],
        "returncode": result.returncode,
        "wall_seconds": elapsed,
        "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
        "stdout_path": str((output / "stdout.txt").relative_to(ROOT)),
        "stderr_path": str((output / "stderr.txt").relative_to(ROOT)),
    }


def _require_command_success(result: dict[str, Any], *, stage: str, output_root: Path) -> None:
    """Stop the dependent rehearsal chain while preserving the failed command record."""
    if result.get("returncode") == 0:
        return
    _json(
        output_root / "integration_failure.json",
        {
            "schema": "reduced_g3_integration_failure_v1",
            "status": "FAIL",
            "failed_stage": stage,
            "command": result,
        },
    )
    raise RuntimeError(f"{stage} failed; dependent rehearsal commands were not launched")


def _checkpoint_state(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metrics = path.parent / "metrics.jsonl"
    batch_hashes: list[str] = []
    if metrics.is_file():
        for line in metrics.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if "train_batch_reference_hash" in record:
                    batch_hashes.append(str(record["train_batch_reference_hash"]))
    model_hash = hashlib.sha256()
    for name, tensor in sorted(payload["model"].items()):
        model_hash.update(name.encode())
        model_hash.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    optimizer_bytes = io.BytesIO()
    torch.save(payload["optimizer"], optimizer_bytes)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "updates_completed": int(payload["counters"]["updates_completed"]),
        "consumed_loss_tokens": int(payload["counters"]["consumed_loss_tokens"]),
        "schedule_cursor": int(payload["schedule_cursor"]),
        "step": int(payload["step"]),
        "run_id": str(payload["run_id"]),
        "runner_identity_hash": payload.get("training_args", {}).get("runner_identity_hash"),
        "model_state_sha256": model_hash.hexdigest(),
        "optimizer_state_sha256": hashlib.sha256(optimizer_bytes.getvalue()).hexdigest(),
        "ordered_batch_hashes": batch_hashes,
        "phase_timing": json.loads((path.parent / "phase_timing.json").read_text(encoding="utf-8"))
        if (path.parent / "phase_timing.json").is_file()
        else None,
    }


def _compare_states(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    fields = ("updates_completed", "consumed_loss_tokens", "schedule_cursor", "step", "run_id", "model_state_sha256", "optimizer_state_sha256", "ordered_batch_hashes")
    checks = {field: left.get(field) == right.get(field) for field in fields}
    return {"checks": checks, "all_equal": all(checks.values())}


def _state_differences(left: Any, right: Any, path: str = "state") -> list[str]:
    """Return exact durable-state differences, including nested tensors and RNG arrays."""
    if isinstance(left, torch.Tensor):
        return [] if isinstance(right, torch.Tensor) and torch.equal(left, right) else [path]
    if isinstance(left, np.ndarray):
        return [] if isinstance(right, np.ndarray) and np.array_equal(left, right) else [path]
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            return [path]
        differences: list[str] = []
        for key in left:
            if path == "state.training_args" and key in {
                "run_dir", "resume", "stop_after_updates", "stop_after_training_seconds",
                "runner_identity", "runner_identity_hash",
            }:
                continue
            differences.extend(_state_differences(left[key], right[key], f"{path}.{key}"))
        return differences
    if isinstance(left, (tuple, list)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            return [path]
        differences = []
        for index, (item_left, item_right) in enumerate(zip(left, right)):
            differences.extend(_state_differences(item_left, item_right, f"{path}[{index}]"))
        return differences
    return [] if left == right else [path]


def _compare_durable_payloads(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = torch.load(left_path, map_location="cpu", weights_only=False)
    right = torch.load(right_path, map_location="cpu", weights_only=False)
    differences = _state_differences(left, right)
    return {
        "all_equal": not differences,
        "compared_fields": sorted(left.keys()) if isinstance(left, dict) else [],
        "excluded_training_args": [
            "run_dir", "resume", "stop_after_updates", "stop_after_training_seconds",
            "runner_identity", "runner_identity_hash",
        ],
        "differences": differences,
    }


def _compare_runner_manifests(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = json.loads(left_path.read_text(encoding="utf-8"))
    right = json.loads(right_path.read_text(encoding="utf-8"))
    left_semantic = dict(left)
    right_semantic = dict(right)
    left_semantic.pop("command", None)
    right_semantic.pop("command", None)
    return {
        "semantic_equal": left_semantic == right_semantic,
        "run_id_equal": left.get("run_id") == right.get("run_id"),
        "excluded_fields": ["command"],
        "left_sha256": sha256(left_path),
        "right_sha256": sha256(right_path),
    }


def _corrupt_copy(source: Path, destination: Path) -> None:
    shutil.copy2(source, destination)
    manifest = source.with_name(source.name + ".manifest.json")
    shutil.copy2(manifest, destination.with_name(destination.name + ".manifest.json"))
    data = bytearray(destination.read_bytes())
    if not data:
        raise RuntimeError("cannot corrupt an empty checkpoint")
    offset = max(0, len(data) // 2)
    data[offset] ^= 0x01
    destination.write_bytes(data)


def _early_export_rejection(run_dir: Path, checkpoint: Path, output: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/run_reduced_baseline.py",
        "verify",
        "--run-dir",
        str(run_dir),
        "--checkpoint",
        str(checkpoint),
    ]
    result = _run(command, output=output)
    marker = "early-stopped and is not export eligible"
    result["rejected"] = result["returncode"] != 0 and marker in (
        (output / "stdout.txt").read_text(encoding="utf-8")
        + (output / "stderr.txt").read_text(encoding="utf-8")
    )
    result["reason"] = marker if result["rejected"] else "UNEXPECTED_ACCEPTANCE_OR_WRONG_FAILURE"
    return result


def _positive_fixture(output: Path) -> dict[str, Any]:
    """Run the already isolated tiny completed-decay fixture through pytest."""
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_reduced_baseline_runner.py",
        "-k",
        "completed_durable_fixture_passes_runner_verify_and_export",
    ]
    result = _run(command, output=output)
    result["status"] = "PASS" if result["returncode"] == 0 else "FAIL"
    result["identity"] = "fixture-scale completed-decay runner identity (test-owned temporary directory)"
    result["production_baseline_claim"] = False
    return result


def _load_metric_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _saved_command_checks(
    record: Any,
    *,
    output_root: Path,
    expected_returncode: int | None,
    required_text: str | None = None,
) -> dict[str, bool]:
    """Validate command results against retained output text, never report labels."""
    checks = {
        "record_is_mapping": isinstance(record, dict),
        "returncode_matches": False,
        "wall_seconds_positive": False,
        "stdout_in_output_root": False,
        "stderr_in_output_root": False,
        "stdout_hash_matches": False,
        "stderr_hash_matches": False,
        "required_text_present": required_text is None,
    }
    if not isinstance(record, dict):
        return checks
    checks["returncode_matches"] = (
        isinstance(record.get("returncode"), int)
        and (expected_returncode is None or record["returncode"] == expected_returncode)
    )
    wall = record.get("wall_seconds")
    checks["wall_seconds_positive"] = isinstance(wall, (int, float)) and not isinstance(wall, bool) and wall > 0
    texts: dict[str, str] = {}
    for stream in ("stdout", "stderr"):
        try:
            path = Path(str(record[f"{stream}_path"]))
            if not path.is_absolute():
                path = ROOT / path
            path = path.resolve()
            path.relative_to(output_root.resolve())
            checks[f"{stream}_in_output_root"] = path.is_file()
            if path.is_file():
                # _run hashes decoded subprocess text. read_text applies the same
                # universal-newline normalization to Windows-persisted CRLF bytes.
                texts[stream] = path.read_text(encoding="utf-8")
                checks[f"{stream}_hash_matches"] = (
                    hashlib.sha256(texts[stream].encode()).hexdigest() == record.get(f"{stream}_sha256")
                )
        except (KeyError, OSError, ValueError):
            continue
    if required_text is not None:
        checks["required_text_present"] = required_text in (texts.get("stdout", "") + texts.get("stderr", ""))
    return checks


def _training_command_checks(
    record: Any, *, run_dir: Path, stop_after_updates: int, resume: Path | None
) -> dict[str, bool]:
    command = record.get("command") if isinstance(record, dict) else None
    checks = {"shape": isinstance(command, list) and len(command) >= 8}
    if not checks["shape"]:
        checks.update({"runner": False, "launch_execute": False, "run_dir": False, "stop": False, "resume": False})
        return checks

    def argument(flag: str) -> str | None:
        try:
            return str(command[command.index(flag) + 1])
        except (ValueError, IndexError):
            return None

    runner = Path(str(command[1]))
    if not runner.is_absolute():
        runner = ROOT / runner
    observed_run_dir = argument("--run-dir")
    observed_resume = argument("--resume")
    checks.update(
        {
            "runner": runner.resolve() == (ROOT / "scripts/run_reduced_baseline.py").resolve(),
            "launch_execute": command[2:4] == ["launch", "--execute"],
            "run_dir": observed_run_dir is not None and Path(observed_run_dir).resolve() == run_dir.resolve(),
            "stop": argument("--stop-after-updates") == str(stop_after_updates),
            "resume": (
                observed_resume is None if resume is None
                else observed_resume is not None and Path(observed_resume).resolve() == resume.resolve()
            ),
        }
    )
    return checks


def _counter_checks(
    state: dict[str, Any], *, updates: int, sequences_per_update: int, loss_tokens_per_update: int
) -> dict[str, bool]:
    return {
        "updates_completed": state.get("updates_completed") == updates,
        "consumed_loss_tokens": state.get("consumed_loss_tokens") == updates * loss_tokens_per_update,
        "schedule_cursor": state.get("schedule_cursor") == updates * sequences_per_update,
        "step": state.get("step") == updates - 1,
    }


def _metric_checks(
    rows: list[dict[str, Any]],
    *,
    expected_hashes: list[str],
    sequences_per_update: int,
    loss_tokens_per_update: int,
    run_id: str,
    schedule_hash: str,
) -> dict[str, bool]:
    return {
        "row_count": len(rows) == len(expected_hashes),
        "update_indices": [row.get("update_index") for row in rows] == list(range(len(expected_hashes))),
        "steps": [row.get("step") for row in rows] == list(range(len(expected_hashes))),
        "loss_tokens": [row.get("consumed_loss_tokens") for row in rows]
        == [(index + 1) * loss_tokens_per_update for index in range(len(expected_hashes))],
        "schedule_cursors": [row.get("schedule_cursor") for row in rows]
        == [(index + 1) * sequences_per_update for index in range(len(expected_hashes))],
        "ordered_batch_hashes": [row.get("train_batch_reference_hash") for row in rows] == expected_hashes,
        "run_ids": all(row.get("run_id") == run_id for row in rows),
        "schedule_hashes": all(row.get("schedule_content_hash") == schedule_hash for row in rows),
    }


def _phase_timing_checks(value: Any, *, validation_expected: bool) -> dict[str, bool]:
    required = (
        "training_optimizer_seconds", "validation_seconds", "checkpoint_seconds", "process_wall_seconds"
    )
    checks = {"is_mapping": isinstance(value, dict)}
    if not isinstance(value, dict):
        checks.update({key: False for key in required})
        checks["process_covers_phases"] = False
        checks["validation_expectation"] = False
        return checks
    for key in required:
        item = value.get(key)
        minimum_inclusive = key == "validation_seconds"
        checks[key] = (
            isinstance(item, (int, float)) and not isinstance(item, bool)
            and (item >= 0 if minimum_inclusive else item > 0)
        )
    checks["process_covers_phases"] = all(checks[key] for key in required) and value[
        "process_wall_seconds"
    ] >= sum(value[key] for key in required[:-1])
    checks["validation_expectation"] = checks["validation_seconds"] and (
        value["validation_seconds"] > 0 if validation_expected else value["validation_seconds"] == 0
    )
    return checks


def _timing_history_record_checks(
    record: Any,
    *,
    invocation_index: int,
    started_at_update: int,
    completed_updates: int,
    resume_checkpoint: Path | None,
    timing: Any,
) -> dict[str, bool]:
    checks = {"is_mapping": isinstance(record, dict), "timing_is_mapping": isinstance(timing, dict)}
    if not isinstance(record, dict) or not isinstance(timing, dict):
        checks.update({"schema": False, "index": False, "bounds": False, "resume": False, "timing_matches": False})
        return checks
    observed_resume = record.get("resume_checkpoint")
    checks.update(
        {
            "schema": record.get("schema") == "training_phase_timing_invocation_v1",
            "index": record.get("invocation_index") == invocation_index,
            "bounds": record.get("started_at_update") == started_at_update
            and record.get("completed_updates") == completed_updates,
            "resume": (
                observed_resume is None if resume_checkpoint is None
                else observed_resume is not None and Path(str(observed_resume)).resolve() == resume_checkpoint.resolve()
            ),
            "timing_matches": all(record.get(key) == value for key, value in timing.items()),
        }
    )
    return checks


def _all_checks(checks: dict[str, Any]) -> bool:
    """Require a nonempty recursive tree whose leaves are literally true."""
    if not checks:
        return False
    return all(_all_checks(value) if isinstance(value, dict) else value is True for value in checks.values())


def run(*, output_root: Path, max_updates: int = 8, interrupt_at: int = 4) -> dict[str, Any]:
    if not 1 <= interrupt_at < max_updates <= 8:
        raise ValueError("section-6 rehearsal requires 1 <= interrupt-at < max-updates <= 8")
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing integration evidence: {output_root}")
    output_root.mkdir(parents=True)
    source_manifest_path = output_root / "source_manifest.json"
    _json(source_manifest_path, _source_manifest())
    uninterrupted = output_root / "uninterrupted"
    resumed = output_root / "interrupted-resumed"
    identity_started = time.perf_counter()
    identity = prepare(
        run_dir=uninterrupted,
        environment={"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda},
    )
    preparation_seconds = time.perf_counter() - identity_started
    _json(output_root / "runner_identity.json", identity)

    base = [sys.executable, "scripts/run_reduced_baseline.py", "launch", "--execute"]
    uninterrupted_result = _run(
        [*base, "--run-dir", str(uninterrupted), "--stop-after-updates", str(max_updates)],
        output=output_root / "uninterrupted-command",
    )
    _require_command_success(uninterrupted_result, stage="uninterrupted", output_root=output_root)
    first_result = _run(
        [*base, "--run-dir", str(resumed), "--stop-after-updates", str(interrupt_at)],
        output=output_root / "interrupted-command",
    )
    _require_command_success(first_result, stage="interrupted", output_root=output_root)
    if (resumed / "phase_timing.json").is_file():
        shutil.copy2(resumed / "phase_timing.json", output_root / "interrupted-command" / "phase_timing.json")
    if (resumed / "phase_timing_history.jsonl").is_file():
        shutil.copy2(
            resumed / "phase_timing_history.jsonl",
            output_root / "interrupted-command" / "phase_timing_history.jsonl",
        )
    interrupted_checkpoint = resumed / "latest.pt"
    interrupted_snapshot_dir = output_root / "interrupted-snapshot"
    interrupted_snapshot = interrupted_snapshot_dir / "latest.pt"
    interrupted_snapshot_manifest = interrupted_snapshot_dir / "latest.pt.manifest.json"
    interrupted_snapshot_dir.mkdir(parents=True, exist_ok=True)
    if interrupted_checkpoint.is_file() and interrupted_checkpoint.with_name(
        interrupted_checkpoint.name + ".manifest.json"
    ).is_file():
        shutil.copy2(interrupted_checkpoint, interrupted_snapshot)
        shutil.copy2(
            interrupted_checkpoint.with_name(interrupted_checkpoint.name + ".manifest.json"),
            interrupted_snapshot_manifest,
        )
    interrupted_state = _checkpoint_state(interrupted_snapshot) if interrupted_snapshot.is_file() else None
    resume_result = _run(
        [
            *base,
            "--run-dir",
            str(resumed),
            "--resume",
            str(resumed / "latest.pt"),
            "--stop-after-updates",
            str(max_updates),
        ],
        output=output_root / "resumed-command",
    )
    _require_command_success(resume_result, stage="resumed", output_root=output_root)

    uninterrupted_checkpoint = uninterrupted / "latest.pt"
    resumed_checkpoint = resumed / "latest.pt"
    if not uninterrupted_checkpoint.is_file() or not resumed_checkpoint.is_file():
        raise RuntimeError("bounded training did not produce both durable latest checkpoints")
    durability_reports = {
        "uninterrupted_latest": verify_checkpoint(uninterrupted_checkpoint).to_dict(),
        "interrupted_snapshot": verify_checkpoint(interrupted_snapshot).to_dict()
        if interrupted_snapshot.is_file()
        else {"ok": False, "results": []},
        "resumed_latest": verify_checkpoint(resumed_checkpoint).to_dict(),
    }
    durability_ok = all(report.get("ok") is True for report in durability_reports.values())
    uninterrupted_state = _checkpoint_state(uninterrupted_checkpoint)
    resumed_state = _checkpoint_state(resumed_checkpoint)
    uninterrupted_manifest = json.loads((uninterrupted / "runner_identity.json").read_text(encoding="utf-8"))
    resumed_manifest = json.loads((resumed / "runner_identity.json").read_text(encoding="utf-8"))
    uninterrupted_run_config = json.loads((uninterrupted / "run_config.json").read_text(encoding="utf-8"))
    resumed_run_config = json.loads((resumed / "run_config.json").read_text(encoding="utf-8"))
    actual_identity_binding = {
        "expected_run_id": identity["run_id"],
        "uninterrupted_run_id": uninterrupted_manifest["run_id"],
        "resumed_run_id": resumed_manifest["run_id"],
        "uninterrupted_checkpoint_run_id": uninterrupted_state["run_id"],
        "interrupted_checkpoint_run_id": interrupted_state["run_id"] if interrupted_state else None,
        "resumed_checkpoint_run_id": resumed_state["run_id"],
        "uninterrupted_run_config_run_id": uninterrupted_run_config["run_id"],
        "resumed_run_config_run_id": resumed_run_config["run_id"],
    }
    actual_identity_binding["outer_manifest_bound"] = (
        actual_identity_binding["expected_run_id"]
        == actual_identity_binding["uninterrupted_run_id"]
        == actual_identity_binding["resumed_run_id"]
    )
    actual_identity_binding["training_payload_bound"] = (
        actual_identity_binding["uninterrupted_checkpoint_run_id"]
        == actual_identity_binding["interrupted_checkpoint_run_id"]
        == actual_identity_binding["resumed_checkpoint_run_id"]
        == actual_identity_binding["uninterrupted_run_config_run_id"]
        == actual_identity_binding["resumed_run_config_run_id"]
    )
    actual_identity_binding["all_bound"] = (
        actual_identity_binding["outer_manifest_bound"]
        and actual_identity_binding["training_payload_bound"]
    )
    uninterrupted_identity_path = uninterrupted / "runner_identity.json"
    resumed_identity_path = resumed / "runner_identity.json"
    checkpoint_identity_binding = {
        "output_copy_semantic_matches_uninterrupted": json.loads(
            (output_root / "runner_identity.json").read_text(encoding="utf-8")
        ) == uninterrupted_manifest,
        "uninterrupted_latest_matches_manifest": uninterrupted_state["runner_identity_hash"]
        == sha256(uninterrupted_identity_path),
        "interrupted_snapshot_matches_manifest": interrupted_state is not None
        and interrupted_state["runner_identity_hash"] == sha256(resumed_identity_path),
        "resumed_latest_matches_manifest": resumed_state["runner_identity_hash"]
        == sha256(resumed_identity_path),
    }
    checkpoint_identity_binding["all_bound"] = all(checkpoint_identity_binding.values())
    comparison = _compare_states(uninterrupted_state, resumed_state)
    durable_comparison = _compare_durable_payloads(uninterrupted_checkpoint, resumed_checkpoint)
    identity_comparison = _compare_runner_manifests(
        uninterrupted / "runner_identity.json", resumed / "runner_identity.json"
    )
    comparison["durable_payload"] = durable_comparison
    comparison["runner_identity"] = identity_comparison
    comparison["all_equal"] = (
        comparison["all_equal"] and durable_comparison["all_equal"] and identity_comparison["semantic_equal"]
    )

    corrupt_path = output_root / "corrupt-latest.pt"
    _corrupt_copy(uninterrupted_checkpoint, corrupt_path)
    corrupt_report = verify_checkpoint(corrupt_path)
    corruption_reasons = [result.reason for result in corrupt_report.failures]
    checksum_rejected = any("CHECKPOINT_CHECKSUM_MISMATCH" in reason for reason in corruption_reasons)
    corruption = {
        "path": str(corrupt_path.relative_to(ROOT)),
        "rejected": checksum_rejected,
        "status": "PASS" if checksum_rejected else "FAIL",
        "required_reason": "CHECKPOINT_CHECKSUM_MISMATCH",
        "failure_reasons": corruption_reasons,
        "report": corrupt_report.to_dict(),
    }
    early_export = _early_export_rejection(
        uninterrupted,
        uninterrupted_checkpoint,
        output_root / "early-export-rejection",
    )
    fixture = _positive_fixture(output_root / "completed-decay-fixture")
    report = {
        "schema": "reduced_g3_integration_evidence_v1",
        "status": "PASS" if (
            uninterrupted_result["returncode"] == 0
            and first_result["returncode"] == 0
            and resume_result["returncode"] == 0
            and durability_ok
            and actual_identity_binding["all_bound"]
            and checkpoint_identity_binding["all_bound"]
            and comparison["all_equal"]
            and corruption["rejected"]
            and early_export["rejected"]
            and fixture["status"] == "PASS"
        ) else "FAIL",
        "claim": "bounded real-shard recovery rehearsal only; canonical G3/G4/G5/G6 remain NOT_RUN",
        "production_horizon_updates": int(identity["total_updates"]),
        "bounded_stop_updates": max_updates,
        "interruption_stop_updates": interrupt_at,
        "run_identity": identity["run_id"],
        "runner_identity_sha256": sha256(output_root / "runner_identity.json"),
        "source_manifest_sha256": sha256(source_manifest_path),
        "actual_identity_binding": actual_identity_binding,
        "checkpoint_identity_binding": checkpoint_identity_binding,
        "durability_verification": {"all_ok": durability_ok, "reports": durability_reports},
        "preparation_seconds": preparation_seconds,
        "uninterrupted": {"command": uninterrupted_result, "checkpoint": uninterrupted_state},
        "interrupted": {"command": first_result, "checkpoint": interrupted_state},
        "resumed": {"command": resume_result, "checkpoint": resumed_state},
        "exact_recovery_comparison": comparison,
        "corruption_rejection": corruption,
        "early_export_rejection": early_export,
        "completed_decay_fixture": fixture,
        "phase_timing": {
            "uninterrupted": uninterrupted_state["phase_timing"],
            "resumed_invocation": resumed_state["phase_timing"],
            "evaluation_seconds": "NOT_RUN_IN_SECTION_6; section-5 CPU smoke remains separate",
            "preparation_seconds": preparation_seconds,
            "total_wall_seconds": uninterrupted_result["wall_seconds"] + first_result["wall_seconds"] + resume_result["wall_seconds"],
        },
        "limitations": [
            "The early-stop checkpoints intentionally fail baseline export eligibility.",
            "This rehearsal does not complete the production horizon or produce a baseline export.",
            "Evaluation is not rerun; section-5 smoke evidence remains separately bound.",
        ],
    }
    _json(output_root / "integration_report.json", report)
    return report


def postverify(output_root: Path) -> dict[str, Any]:
    """Independently audit retained artifacts and emit a fail-closed successor."""
    output_root = output_root.resolve()
    source_report_path = output_root / "integration_report.json"
    successor_path = output_root / "integration_report.postverified.json"
    result: dict[str, Any] = {
        "schema": "reduced_g3_integration_postverification_v1",
        "status": "FAIL",
        "claim": "bounded real-shard recovery rehearsal only; canonical G3/G4/G5/G6 remain NOT_RUN",
        "source_report": {
            "path": str(source_report_path.relative_to(ROOT)),
            "sha256": sha256(source_report_path) if source_report_path.is_file() else None,
            "trusted_for_decision": False,
        },
        "postverifier_source_sha256": sha256(Path(__file__)),
        "checks": {},
    }
    try:
        source = json.loads(source_report_path.read_text(encoding="utf-8"))
        uninterrupted = output_root / "uninterrupted"
        resumed = output_root / "interrupted-resumed"
        snapshot = output_root / "interrupted-snapshot" / "latest.pt"
        source_manifest_path = output_root / "source_manifest.json"
        identity_path = output_root / "runner_identity.json"
        uninterrupted_identity_path = uninterrupted / "runner_identity.json"
        resumed_identity_path = resumed / "runner_identity.json"
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        uninterrupted_manifest = json.loads(uninterrupted_identity_path.read_text(encoding="utf-8"))
        current_identity = prepare(
            run_dir=uninterrupted,
            environment={"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda},
        )
        recorded_source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        current_source_manifest = _source_manifest()
        uninterrupted_checkpoint = uninterrupted / "latest.pt"
        resumed_checkpoint = resumed / "latest.pt"
        uninterrupted_state = _checkpoint_state(uninterrupted_checkpoint)
        interrupted_state = _checkpoint_state(snapshot)
        resumed_state = _checkpoint_state(resumed_checkpoint)
        uninterrupted_run_config = json.loads((uninterrupted / "run_config.json").read_text(encoding="utf-8"))
        resumed_run_config = json.loads((resumed / "run_config.json").read_text(encoding="utf-8"))
        max_updates = int(source["bounded_stop_updates"])
        interrupt_at = int(source["interruption_stop_updates"])
        loss_tokens_per_update = int(current_identity["loss_tokens_per_update"])
        sequences_per_update = loss_tokens_per_update // int(current_identity["sequence_length"])
        exposure = load_exposure_plan(
            ROOT / current_identity["paths"]["exposure_plan"],
            (ROOT / current_identity["paths"]["component_1"], ROOT / current_identity["paths"]["component_2"]),
        )
        entries = exposure.components[0].entries + exposure.components[1].entries
        expected_hashes = [
            training_order_hash(entries[index * sequences_per_update:(index + 1) * sequences_per_update])
            for index in range(max_updates)
        ]
        uninterrupted_rows = _load_metric_rows(uninterrupted / "metrics.jsonl")
        resumed_rows = _load_metric_rows(resumed / "metrics.jsonl")
        resumed_manifest = json.loads(resumed_identity_path.read_text(encoding="utf-8"))
        runner_comparison = _compare_runner_manifests(uninterrupted_identity_path, resumed_identity_path)
        durability = {
            "uninterrupted_latest": verify_checkpoint(uninterrupted_checkpoint).to_dict(),
            "interrupted_snapshot": verify_checkpoint(snapshot).to_dict(),
            "resumed_latest": verify_checkpoint(resumed_checkpoint).to_dict(),
        }
        corrupt_report = verify_checkpoint(output_root / "corrupt-latest.pt")
        corrupt_reasons = [item.reason for item in corrupt_report.failures]
        exact_summary = _compare_states(uninterrupted_state, resumed_state)
        durable_comparison = _compare_durable_payloads(uninterrupted_checkpoint, resumed_checkpoint)
        outer_ids = [
            current_identity.get("run_id"), identity.get("run_id"),
            uninterrupted_manifest.get("run_id"), resumed_manifest.get("run_id"),
        ]
        training_ids = [
            uninterrupted_state.get("run_id"), interrupted_state.get("run_id"), resumed_state.get("run_id"),
            uninterrupted_run_config.get("run_id"), resumed_run_config.get("run_id"),
        ]
        interrupted_phase_path = output_root / "interrupted-command" / "phase_timing.json"
        interrupted_phase = (
            json.loads(interrupted_phase_path.read_text(encoding="utf-8"))
            if interrupted_phase_path.is_file() else None
        )
        uninterrupted_history = _load_metric_rows(uninterrupted / "phase_timing_history.jsonl")
        interrupted_history = _load_metric_rows(output_root / "interrupted-command" / "phase_timing_history.jsonl")
        resumed_history = _load_metric_rows(resumed / "phase_timing_history.jsonl")
        checks: dict[str, Any] = {
            "source_shape": {
                "schema": source.get("schema") == "reduced_g3_integration_evidence_v1",
                "fixed_horizon": source.get("production_horizon_updates") == current_identity.get("total_updates") == 3815,
                "bounded_stops": max_updates == 8 and interrupt_at == 4,
                "claim_is_bounded": source.get("claim") == result["claim"],
            },
            "current_input_identity": {
                "output_copy_matches": identity == current_identity,
                "uninterrupted_matches": uninterrupted_manifest == current_identity,
                "resumed_semantic_matches": runner_comparison["semantic_equal"],
                "outer_ids_match": len(set(outer_ids)) == 1,
                "exposure_hash_matches": exposure.content_hash == current_identity.get("exposure_content_hash"),
            },
            "source_custody": {
                "manifest_matches_current_tree": recorded_source_manifest == current_source_manifest,
                "source_report_binds_manifest": source.get("source_manifest_sha256") == sha256(source_manifest_path),
            },
            "checkpoint_identity": {
                "training_ids_match": len(set(training_ids)) == 1,
                "uninterrupted_manifest_hash": uninterrupted_state.get("runner_identity_hash")
                == sha256(uninterrupted_identity_path),
                "interrupted_manifest_hash": interrupted_state.get("runner_identity_hash")
                == sha256(resumed_identity_path),
                "resumed_manifest_hash": resumed_state.get("runner_identity_hash") == sha256(resumed_identity_path),
            },
            "durability": {name: report.get("ok") is True for name, report in durability.items()},
            "counters": {
                "uninterrupted": _counter_checks(
                    uninterrupted_state, updates=max_updates, sequences_per_update=sequences_per_update,
                    loss_tokens_per_update=loss_tokens_per_update,
                ),
                "interrupted": _counter_checks(
                    interrupted_state, updates=interrupt_at, sequences_per_update=sequences_per_update,
                    loss_tokens_per_update=loss_tokens_per_update,
                ),
                "resumed": _counter_checks(
                    resumed_state, updates=max_updates, sequences_per_update=sequences_per_update,
                    loss_tokens_per_update=loss_tokens_per_update,
                ),
            },
            "ordered_input": {
                "uninterrupted": _metric_checks(
                    uninterrupted_rows, expected_hashes=expected_hashes, sequences_per_update=sequences_per_update,
                    loss_tokens_per_update=loss_tokens_per_update, run_id=str(training_ids[0]),
                    schedule_hash=exposure.content_hash,
                ),
                "resumed": _metric_checks(
                    resumed_rows, expected_hashes=expected_hashes, sequences_per_update=sequences_per_update,
                    loss_tokens_per_update=loss_tokens_per_update, run_id=str(training_ids[0]),
                    schedule_hash=exposure.content_hash,
                ),
                "interrupted_prefix_retained": [row.get("train_batch_reference_hash") for row in resumed_rows[:interrupt_at]]
                == expected_hashes[:interrupt_at],
            },
            "exact_recovery": {
                "summary": exact_summary["all_equal"],
                "durable_payload": durable_comparison["all_equal"],
                "runner_semantics": runner_comparison["semantic_equal"],
            },
            "commands": {
                "uninterrupted": _saved_command_checks(
                    source.get("uninterrupted", {}).get("command"), output_root=output_root, expected_returncode=0
                ),
                "interrupted": _saved_command_checks(
                    source.get("interrupted", {}).get("command"), output_root=output_root, expected_returncode=0
                ),
                "resumed": _saved_command_checks(
                    source.get("resumed", {}).get("command"), output_root=output_root, expected_returncode=0
                ),
                "early_export": _saved_command_checks(
                    source.get("early_export_rejection"), output_root=output_root, expected_returncode=1,
                    required_text="early-stopped and is not export eligible",
                ),
                "completed_decay_fixture": _saved_command_checks(
                    source.get("completed_decay_fixture"), output_root=output_root, expected_returncode=0,
                    required_text="1 passed",
                ),
                "uninterrupted_semantics": _training_command_checks(
                    source.get("uninterrupted", {}).get("command"), run_dir=uninterrupted,
                    stop_after_updates=max_updates, resume=None,
                ),
                "interrupted_semantics": _training_command_checks(
                    source.get("interrupted", {}).get("command"), run_dir=resumed,
                    stop_after_updates=interrupt_at, resume=None,
                ),
                "resumed_semantics": _training_command_checks(
                    source.get("resumed", {}).get("command"), run_dir=resumed,
                    stop_after_updates=max_updates, resume=resumed_checkpoint,
                ),
            },
            "negative_and_positive_controls": {
                "corrupt_checksum_rejected": (not corrupt_report.ok)
                and any("CHECKPOINT_CHECKSUM_MISMATCH" in reason for reason in corrupt_reasons),
                "fixture_is_nonproduction": source.get("completed_decay_fixture", {}).get("production_baseline_claim") is False,
                "early_checkpoint_is_incomplete": uninterrupted_state.get("updates_completed")
                < current_identity.get("total_updates"),
            },
            "phase_timing": {
                "uninterrupted": _phase_timing_checks(uninterrupted_state.get("phase_timing"), validation_expected=True),
                "interrupted": _phase_timing_checks(interrupted_phase, validation_expected=True),
                "resumed": _phase_timing_checks(resumed_state.get("phase_timing"), validation_expected=False),
                "uninterrupted_history_count": len(uninterrupted_history) == 1,
                "interrupted_history_count": len(interrupted_history) == 1,
                "resumed_history_count": len(resumed_history) == 2,
                "uninterrupted_history": _timing_history_record_checks(
                    uninterrupted_history[0] if len(uninterrupted_history) == 1 else None,
                    invocation_index=0, started_at_update=0, completed_updates=max_updates,
                    resume_checkpoint=None, timing=uninterrupted_state.get("phase_timing"),
                ),
                "interrupted_history": _timing_history_record_checks(
                    interrupted_history[0] if len(interrupted_history) == 1 else None,
                    invocation_index=0, started_at_update=0, completed_updates=interrupt_at,
                    resume_checkpoint=None, timing=interrupted_phase,
                ),
                "resumed_history_first_record": (
                    len(resumed_history) == 2 and len(interrupted_history) == 1
                    and resumed_history[0] == interrupted_history[0]
                ),
                "resumed_history": _timing_history_record_checks(
                    resumed_history[1] if len(resumed_history) == 2 else None,
                    invocation_index=1, started_at_update=interrupt_at, completed_updates=max_updates,
                    resume_checkpoint=resumed_checkpoint, timing=resumed_state.get("phase_timing"),
                ),
            },
        }
        result["checks"] = checks
        result["status"] = "PASS" if _all_checks(checks) else "FAIL"
        result["identities"] = {"runner_run_id": outer_ids[0], "training_run_id": training_ids[0]}
        result["artifacts"] = {
            "runner_identity": {"path": str(identity_path.relative_to(ROOT)), "sha256": sha256(identity_path)},
            "source_manifest": {"path": str(source_manifest_path.relative_to(ROOT)), "sha256": sha256(source_manifest_path)},
            "uninterrupted_checkpoint": uninterrupted_state,
            "interrupted_checkpoint": interrupted_state,
            "resumed_checkpoint": resumed_state,
            "durability_reports": durability,
            "exact_recovery": {
                "summary": exact_summary,
                "durable_payload": durable_comparison,
                "runner_identity": runner_comparison,
            },
            "corruption_failure_reasons": corrupt_reasons,
            "expected_ordered_batch_hashes": expected_hashes,
            "interrupted_phase_timing_path": (
                str(interrupted_phase_path.relative_to(ROOT)) if interrupted_phase_path.is_file() else "MISSING"
            ),
        }
    # Any load, parse, or comparison failure is evidence failure and must still leave a
    # machine-readable successor rather than aborting before the FAIL can be recorded.
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    _json(successor_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-updates", type=int, default=8)
    parser.add_argument("--interrupt-at", type=int, default=4)
    parser.add_argument("--postverify", action="store_true", help="independently audit retained artifacts")
    args = parser.parse_args()
    if args.postverify:
        report = postverify(args.output_root)
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(0 if report["status"] == "PASS" else 1)
    report = run(output_root=args.output_root, max_updates=args.max_updates, interrupt_at=args.interrupt_at)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
