"""Plan or execute a bounded, production-path G4 baseline training profile.

This wrapper never changes the training recipe.  It delegates preparation and command
construction to ``run_reduced_baseline`` and adds only outer accounting, telemetry, a
second wall-clock watchdog, and fail-closed measurement validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch  # noqa: E402
from scripts.run_reduced_baseline import SELECTED_CONFIG as CONFIG, ROOT, launch_command, prepare  # noqa: E402
from scripts.profile_telemetry import sample_telemetry, summarize_telemetry  # noqa: E402
from tinybench_lm.operations import ThroughputMeasurement, throughput_violations  # noqa: E402

WARMUP_UPDATES = 38
TOTAL_UPDATES = 3815
UPDATE_CAP = 800
OPTIMIZER_STOP_SECONDS = 2400.0
MINIMUM_VALID_SECONDS = 1800.0
OUTER_WATCHDOG_SECONDS = 3600.0
TELEMETRY_CADENCE_SECONDS = 5.0


class ProfileError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_number(name: str, value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ProfileError(f"{name} must be a finite measurement")
    result = float(value)
    if result < 0 or (positive and result == 0):
        raise ProfileError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return result


def validate_identity(identity: Mapping[str, Any]) -> None:
    expected = {
        "total_updates": TOTAL_UPDATES,
        "warmup_updates": WARMUP_UPDATES,
        "validation_mode": "full-dev",
        "eval_interval": 100,
        "save_interval": 100,
        "micro_batch_size": 8,
        "gradient_accumulation": 32,
        "sequence_length": 1024,
        "peak_lr": 0.0006,
        "decay_updates": 382,
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ProfileError(f"fixed baseline identity mismatch for {key}: {identity.get(key)!r}")


def build_profile_command(identity: Mapping[str, Any], train_dir: Path) -> list[str]:
    validate_identity(identity)
    command = launch_command(
        identity,
        run_dir=train_dir,
        stop_after_updates=UPDATE_CAP,
        stop_after_training_seconds=OPTIMIZER_STOP_SECONDS,
    )
    required = {
        "--steps": str(TOTAL_UPDATES),
        "--warmup-steps": str(WARMUP_UPDATES),
        "--validation-mode": "full-dev",
        "--eval-interval": "100",
        "--save-interval": "100",
        "--stop-after-updates": str(UPDATE_CAP),
        "--stop-after-training-seconds": str(OPTIMIZER_STOP_SECONDS),
    }
    for flag, value in required.items():
        try:
            actual = command[command.index(flag) + 1]
        except (ValueError, IndexError) as error:
            raise ProfileError(f"profile command is missing {flag}") from error
        if actual != value:
            raise ProfileError(f"profile command has unsafe {flag}={actual!r}")
    return command


def analyze_rows(rows: Sequence[Mapping[str, Any]], *, machine_id: str = "production_gpu") -> dict[str, Any]:
    if not rows:
        raise ProfileError("metrics contain no optimizer updates")
    seen: set[int] = set()
    valid: list[tuple[float, float]] = []
    all_seconds = 0.0
    validation_events: list[int] = []
    allocated_peaks: list[float] = []
    previous_finish = 0.0
    for position, row in enumerate(rows):
        index = row.get("update_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= TOTAL_UPDATES:
            raise ProfileError(f"metrics row {position} has invalid update_index")
        if index in seen:
            raise ProfileError("metrics contain duplicate update_index")
        if index != position:
            raise ProfileError("metrics must be an ordered contiguous prefix from zero")
        seen.add(index)
        seconds = _require_number(f"metrics[{index}].step_seconds", row.get("step_seconds"), positive=True)
        tokens = _require_number(f"metrics[{index}].loss_tokens_per_update", row.get("loss_tokens_per_update"), positive=True)
        if tokens != 262_144:
            raise ProfileError(f"metrics[{index}] does not use the fixed loss-token batch")
        rate = _require_number(f"metrics[{index}].tokens_per_second", row.get("tokens_per_second"), positive=True)
        if not math.isclose(rate, tokens / seconds, rel_tol=1e-6):
            raise ProfileError(f"metrics[{index}] token rate disagrees with tokens/time")
        all_seconds += seconds
        start = _require_number("update_started_seconds", row.get("update_started_seconds"))
        finish = _require_number("optimizer_finished_seconds", row.get("optimizer_finished_seconds"))
        if start < previous_finish or not math.isclose(finish - start, seconds, abs_tol=1e-7):
            raise ProfileError("update timing is inconsistent or nonmonotonic")
        previous_finish = finish
        allocated_peaks.append(_require_number(f"metrics[{index}].peak_vram_gib", row.get("peak_vram_gib")))
        if "validation_loss" in row:
            _require_number("validation_loss", row["validation_loss"])
            for name, expected in {
                "validation_mode": "full-dev", "validation_sequence_count": 753,
                "validation_scored_token_count": 771072, "validation_batch_count": 95,
            }.items():
                if row.get(name) != expected:
                    raise ProfileError(f"full-dev coverage mismatch: {name}")
            validation_events.append(index + 1)
        if index >= WARMUP_UPDATES:
            valid.append((tokens, seconds))
    if seen != set(range(max(seen) + 1)):
        raise ProfileError("metrics updates are not a contiguous prefix from zero")
    if len(seen) > UPDATE_CAP:
        raise ProfileError("metrics exceed the absolute profile update cap")
    expected_validation = [update for update in range(1, max(seen) + 2) if update == 1 or update % 100 == 0]
    if validation_events != expected_validation:
        raise ProfileError(f"full-dev cadence mismatch: observed {validation_events}, expected {expected_validation}")
    if not valid:
        raise ProfileError("no post-warmup optimizer samples")
    window = sum(seconds for _, seconds in valid)
    rates = tuple(tokens / seconds for tokens, seconds in valid)
    measurement = ThroughputMeasurement(machine_id, rates, window, True)
    problems = throughput_violations(measurement)
    if window < MINIMUM_VALID_SECONDS or problems:
        raise ProfileError("invalid sustained window: " + "; ".join(problems or (f"{window}s < {MINIMUM_VALID_SECONDS}s",)))
    if window > 3600:
        raise ProfileError("post-warmup optimizer window exceeds 3600 seconds")
    if all_seconds < OPTIMIZER_STOP_SECONDS and len(seen) != UPDATE_CAP:
        raise ProfileError("training exited without reaching either declared stop boundary")
    if sum(row["step_seconds"] for row in rows[:-1]) >= OPTIMIZER_STOP_SECONDS:
        raise ProfileError("training continued past the first optimizer stop boundary")
    if not any(update >= 100 for update in validation_events):
        raise ProfileError("profile did not include ordinary validation/checkpoint cadence")
    return {
        **measurement.to_dict(),
        "weighted_tokens_per_second": sum(tokens for tokens, _ in valid) / window,
        "optimizer_seconds_all_updates": all_seconds,
        "warmup_updates_excluded": WARMUP_UPDATES,
        "first_included_update_index": min(index for index in seen if index >= WARMUP_UPDATES),
        "full_dev_completed_updates": validation_events,
        "peak_allocated_vram_gib": max(allocated_peaks),
        "elapsed_training_window_seconds": rows[-1]["optimizer_finished_seconds"] - rows[WARMUP_UPDATES]["update_started_seconds"],
        "elapsed_window_definition": "start of update 39 through final optimizer finish; includes intervening validation/saves, excludes trailing saves",
        "stop_reason": "update_cap" if len(rows) == UPDATE_CAP else "optimizer_seconds",
    }


def validate_phase_timing(payload: Mapping[str, Any]) -> dict[str, float]:
    required = ("training_optimizer_seconds", "data_wait_seconds", "validation_seconds", "checkpoint_seconds", "process_wall_seconds")
    result = {name: _require_number(name, payload.get(name), positive=name != "data_wait_seconds") for name in required}
    if result["data_wait_seconds"] > result["training_optimizer_seconds"]:
        raise ProfileError("data wait cannot exceed optimizer timing that contains it")
    if result["process_wall_seconds"] < sum(result[name] for name in ("training_optimizer_seconds", "validation_seconds", "checkpoint_seconds")):
        raise ProfileError("trainer phase seconds exceed trainer process wall")
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def snapshot_inputs(identity: Mapping[str, Any]) -> dict[str, str]:
    paths = set(identity["paths"].values())
    paths.add(identity["paths"].get("baseline_contract", "configs/training/baseline_reduced_v1.yaml"))
    paths.add("data/shards/reduced_5pct_v1/stable_train.manifest.json")
    return {str(path): sha256(ROOT / path) for path in sorted(paths)}


def snapshot_source() -> dict[str, str]:
    paths = set()
    for folder in ("src/tinybench_lm", "scripts", "tests", "configs", "constraints"):
        paths.update(p for p in (ROOT / folder).rglob("*") if p.is_file() and
                     "__pycache__" not in p.parts and p.suffix in {".py", ".yaml", ".json", ".toml", ".txt", ".sha256"})
    paths.update(ROOT / name for name in ("train.py", "evaluate.py", "generate.py", "Dockerfile", ".dockerignore", "pyproject.toml"))
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(paths)}


def _stop_child(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
        process.wait(timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)


def validate_training_artifacts(train_dir: Path, identity: Mapping[str, Any], outer_wall: float) -> dict[str, Any]:
    paths = {name: train_dir / name for name in (
        "metrics.jsonl", "phase_timing.json", "phase_timing_history.jsonl", "runner_identity.json",
        "latest.pt.manifest.json",
    )}
    rows = [json.loads(line) for line in paths["metrics.jsonl"].read_text().splitlines()]
    throughput = analyze_rows(rows)
    phase = validate_phase_timing(json.loads(paths["phase_timing.json"].read_text()))
    history = [json.loads(line) for line in paths["phase_timing_history.jsonl"].read_text().splitlines()]
    if len(history) != 1:
        raise ProfileError("fresh profile must retain exactly one trainer invocation")
    record = history[0]
    expected = {"schema": "training_phase_timing_invocation_v1", "invocation_index": 0,
                "started_at_update": 0, "completed_updates": len(rows), "resume_checkpoint": None}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ProfileError("trainer phase history bounds/identity mismatch")
    if any(record.get(key) != value for key, value in phase.items()):
        raise ProfileError("phase history differs from latest timing")
    if not math.isclose(phase["training_optimizer_seconds"], throughput["optimizer_seconds_all_updates"], rel_tol=1e-9):
        raise ProfileError("phase optimizer seconds differ from metric sum")
    if phase["process_wall_seconds"] > outer_wall or rows[-1]["optimizer_finished_seconds"] > phase["process_wall_seconds"]:
        raise ProfileError("trainer timing exceeds its enclosing wall interval")
    run_id = rows[0].get("run_id")
    invocation_id = record.get("metric_invocation_id")
    if not run_id or not invocation_id:
        raise ProfileError("missing training run/invocation identity")
    for index, row in enumerate(rows):
        if (row.get("run_id") != run_id or row.get("invocation_id") != invocation_id or
            row.get("schedule_content_hash") != identity["exposure_content_hash"] or
            row.get("schedule_cursor") != (index + 1) * 256 or
            row.get("consumed_loss_tokens") != (index + 1) * 262144):
            raise ProfileError("metric identity/cursor/token counter mismatch")
        if "validation_loss" in row and row.get("validation_schedule_content_hash") != identity["validation_schedule_hash"]:
            raise ProfileError("development schedule identity mismatch")
    recorded = json.loads(paths["runner_identity.json"].read_text())
    if recorded != identity:
        raise ProfileError("runner identity drifted during training")
    checkpoint = train_dir / "latest.pt"
    manifest = json.loads(paths["latest.pt.manifest.json"].read_text())
    if (not checkpoint.is_file() or checkpoint.stat().st_size != manifest.get("bytes") or
        manifest.get("updates_completed") != len(rows) or manifest.get("run_id") != run_id):
        raise ProfileError("bounded checkpoint manifest/counter mismatch")
    return {"throughput": throughput, "trainer_phase_timing": phase,
            "trainer_phase_history": history, "training_run_id": run_id,
            "artifact_sha256": {name: sha256(path) for name, path in paths.items()},
            "checkpoint_scope": "manifest/counters/size inspected; durable tensor verification belongs to recovery"}


def execute(output_dir: Path, command: Sequence[str], identity: Mapping[str, Any], *, watchdog_seconds: float = OUTER_WATCHDOG_SECONDS) -> dict[str, Any]:
    train_dir = output_dir / "train"
    train_dir.mkdir()
    _write_json(train_dir / "runner_identity.json", identity)
    stdout_path, stderr_path = output_dir / "stdout.log", output_dir / "stderr.log"
    samples: list[dict[str, Any]] = []
    sampler_errors: list[str] = []
    stop_sampling = threading.Event()
    sampling_failed = threading.Event()
    started_utc, started = utc_now(), time.perf_counter()
    def collect() -> None:
        try:
            with (output_dir / "telemetry.jsonl").open("x", encoding="utf-8") as stream:
                while not stop_sampling.is_set():
                    sample_started = time.perf_counter()
                    sample = sample_telemetry()
                    sample["outer_elapsed_seconds"] = time.perf_counter() - started
                    samples.append(sample)
                    stream.write(json.dumps(sample, allow_nan=False) + "\n")
                    stream.flush()
                    if summarize_telemetry([sample])["status"] != "MEASURED":
                        raise ProfileError("required telemetry collection failed")
                    stop_sampling.wait(max(0.0, TELEMETRY_CADENCE_SECONDS - (time.perf_counter() - sample_started)))
        except Exception as error:
            sampler_errors.append(f"{type(error).__name__}: {error}")
            sampling_failed.set()
    sampler = threading.Thread(target=collect, daemon=True)
    timed_out = False
    process = None
    returncode = None
    failure = None
    # Start/end receipts survive child failure; an interrupted wrapper leaves a RUNNING
    # receipt, which the next operator must reconcile before any new attempt.
    _write_json(output_dir / "active_process.json", {"status": "STARTING", "wrapper_pid": os.getpid(),
                "command": list(command), "started_at_utc": started_utc, "watchdog_seconds": watchdog_seconds})
    try:
        with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            process = subprocess.Popen(list(command), cwd=ROOT, stdout=stdout, stderr=stderr, creationflags=creationflags)
            _write_json(output_dir / "active_process.json", {"status": "RUNNING", "wrapper_pid": os.getpid(),
                        "child_pid": process.pid, "command": list(command), "started_at_utc": started_utc,
                        "watchdog_seconds": watchdog_seconds})
            sampler.start()
            while process.poll() is None:
                if sampling_failed.is_set():
                    raise ProfileError("telemetry sampler failed: " + "; ".join(sampler_errors))
                if time.perf_counter() - started >= watchdog_seconds:
                    timed_out = True
                    raise ProfileError("outer watchdog expired")
                try:
                    process.wait(timeout=min(1.0, max(0.001, watchdog_seconds - (time.perf_counter() - started))))
                except subprocess.TimeoutExpired:
                    pass
            returncode = process.returncode
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        if process is not None:
            try:
                _stop_child(process)
                returncode = process.returncode
            except (OSError, subprocess.SubprocessError) as error:
                failure = f"child termination unconfirmed: {error}"
        wall = time.perf_counter() - started
        ended_utc = utc_now()
        stop_sampling.set()
        if sampler.ident is not None:
            sampler.join(timeout=12)
        if sampler.is_alive() or sampler_errors:
            failure = failure or "telemetry sampler incomplete: " + "; ".join(sampler_errors)
    invocation = {"schema": "g4_profile_outer_invocation_v1", "command": list(command), "cwd": str(ROOT), "started_at_utc": started_utc,
                  "ended_at_utc": ended_utc, "outer_wall_seconds": wall, "returncode": returncode, "watchdog_seconds": watchdog_seconds,
                  "watchdog_expired": timed_out, "stdout_path": str(stdout_path), "stdout_sha256": sha256(stdout_path),
                  "stderr_path": str(stderr_path), "stderr_sha256": sha256(stderr_path), "failure": failure,
                  "child_pid": process.pid if process else None,
                  "telemetry_path": str(output_dir / "telemetry.jsonl"),
                  "telemetry_sha256": sha256(output_dir / "telemetry.jsonl") if (output_dir / "telemetry.jsonl").is_file() and not sampler.is_alive() else None}
    history_path = output_dir / "outer_invocation_history.jsonl"
    with history_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(invocation, sort_keys=True) + "\n")
    _write_json(output_dir / "active_process.json", {"status": "EXITED" if process is None or process.poll() is not None else "TERMINATION_UNCONFIRMED", **invocation})
    report = {"schema": "g4_bounded_profile_v1", "scope": "ENGINEERING_PROFILE_NOT_BASELINE",
              "status": "FAIL", "invocation": invocation}
    try:
        if failure or timed_out or returncode != 0:
            raise ProfileError(f"training command failed (returncode={returncode}, watchdog_expired={timed_out}): {failure}")
        telemetry = summarize_telemetry(samples)
        if telemetry["status"] != "MEASURED":
            raise ProfileError("required production telemetry is unavailable")
        if len(samples) < 3 or samples[0]["outer_elapsed_seconds"] > 15 or wall - samples[-1]["outer_elapsed_seconds"] > 15:
            raise ProfileError("telemetry does not cover the command interval")
        report.update(validate_training_artifacts(train_dir, identity, wall))
        report.update({"status": "MEASURED", "telemetry": telemetry, "runner_run_id": identity["run_id"]})
    except Exception as error:
        report["failure"] = f"{type(error).__name__}: {error}"
        raise ProfileError(report["failure"]) from error
    finally:
        _write_json(output_dir / "measurement.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    engineering = (ROOT / "runs/verification/g4").resolve()
    if not output.is_relative_to(engineering) or output == engineering:
        raise ProfileError("output must be a fresh engineering directory below runs/verification/g4")
    if output.exists():
        raise ProfileError("profile output directory must be fresh and unique")
    output.mkdir(parents=True)
    wrapper_started = time.perf_counter()
    receipt: dict[str, Any] = {"status": "FAIL", "execute_requested": args.execute,
                               "started_at_utc": utc_now(), "stage": "preparation"}
    try:
        preparation_started = time.perf_counter()
        try:
            identity = prepare(config_path=args.config, run_dir=output / "train", environment={
                "python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda})
        finally:
            receipt["preparation_seconds"] = time.perf_counter() - preparation_started
        command = build_profile_command(identity, output / "train")
        # prepare's default unbounded command is replaced in this engineering manifest.
        identity["command"] = command
        receipt["stage"] = "source_input_telemetry_preflight"
        binding_started = time.perf_counter()
        source, inputs = snapshot_source(), snapshot_inputs(identity)
        _write_json(output / "source_manifest.json", source)
        _write_json(output / "input_manifest.json", inputs)
        initial = sample_telemetry()
        telemetry = summarize_telemetry([initial])
        _write_json(output / "telemetry_preflight.json", {"sample": initial, "summary": telemetry})
        receipt["artifact_and_telemetry_preflight_seconds"] = time.perf_counter() - binding_started
        plan = {"schema": "g4_bounded_profile_plan_v1", "scope": "ENGINEERING_PROFILE_NOT_BASELINE", "execute": args.execute,
                "identity": identity, "preparation_seconds": receipt["preparation_seconds"], "command": command,
                "source_manifest_sha256": sha256(output / "source_manifest.json"),
                "input_manifest_sha256": sha256(output / "input_manifest.json"),
                "machine": {"host": platform.node(), "platform": platform.platform(), "environment": identity.get("environment")},
                "telemetry_preflight": telemetry, "limits": {"update_cap": UPDATE_CAP,
                "optimizer_stop_seconds": OPTIMIZER_STOP_SECONDS, "minimum_post_warmup_seconds": MINIMUM_VALID_SECONDS,
                "warmup_updates_excluded": WARMUP_UPDATES, "outer_watchdog_seconds": OUTER_WATCHDOG_SECONDS,
                "interrupt_grace_seconds": 10, "kill_wait_seconds": 10, "telemetry_cadence_seconds": TELEMETRY_CADENCE_SECONDS}}
        _write_json(output / "profile_plan.json", plan)
        if telemetry["status"] != "MEASURED":
            raise ProfileError("required telemetry preflight unavailable; inspect telemetry_preflight.json")
        if args.execute:
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise ProfileError("production profile requires the verified CUDA/BF16 environment")
            receipt["stage"] = "bounded_training_command"
            report = execute(output, command, identity)
            receipt["stage"] = "post_run_custody"
            if snapshot_source() != source or snapshot_inputs(identity) != inputs:
                report.update({"status": "FAIL", "failure": "source or input identity changed during the profile"})
                _write_json(output / "measurement.json", report)
                raise ProfileError(report["failure"])
        receipt.update({"status": "MEASURED" if args.execute else "PLAN_ONLY", "stage": "complete"})
        print(json.dumps({"status": receipt["status"], "run_id": identity["run_id"],
                          "command": command, "profile_plan": str(output / "profile_plan.json")}, indent=2))
    except BaseException as error:
        receipt["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        receipt["ended_at_utc"] = utc_now()
        receipt["wrapper_main_wall_seconds"] = time.perf_counter() - wrapper_started
        receipt["wall_scope"] = "wrapper main entry to receipt; excludes wrapper imports and final interpreter shutdown"
        _write_json(output / "wrapper_receipt.json", receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
